from argparse import ArgumentParser
from functools import partial
from copy import deepcopy
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

import ignite.distributed as idist
import wandb

from datasets import load_datasets
from models import load_backbone
from trainers import collect_features
from utils import Logger, get_engine_mock


def build_step(X, Y, classifier, optimizer, w):
    def step():
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(X), Y, reduction='sum')
        for p in classifier.parameters():
            loss = loss + p.pow(2).sum().mul(w)
        loss.backward()
        return loss
    return step


def compute_accuracy(X, Y, classifier):
    with torch.no_grad():
        preds = classifier(X).argmax(1)
        acc = (preds == Y).float().mean().item()
    return acc


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()


    ckpt_parents = set([Path(c).parent for c in args.ckpt])
    assert len(set(ckpt_parents)) == 1, f"Expected a single checkpoints directory but got {ckpt_parents}"
    if args.logdir is not None:
        logdir = args.logdir
    else:
        logdir = list(ckpt_parents)[0]

    # args.origin_run_name = logdir.name
    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"lin-{args.augm}-{args.dataset}", args=args,
        job_type="predict-augmentation"

    )

    # DATASETS

    datasets = load_datasets(dataset=args.dataset,
                             datadir=args.datadir,
                             pretrain_data=args.pretrain_data)


    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=False)
    valloader   = build_dataloader(datasets['val'],   drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)

    ## TRAIN to fit the data
    options = [-90, 0, 90, 180]
    num_classes = len(options)
    idx_to_class = {i:j for i, j in enumerate(options)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    logger.log_msg(f"Rotating the images - train")
    rotated, angles = [], []
    for i, batch in enumerate(trainloader):
        X, Y = batch
        angle = random.choice(options)
        img = TF.rotate(X, angle)
        #rotated.append(img)
        #angles.append(torch.Tensor([angle] * len(img)))
        current_dataset = torch.utils.data.TensorDataset(torch.Tensor(img), torch.Tensor([class_to_idx[angle]] * len(img)).type(torch.LongTensor))
        if i == 0:
            rotation_dataset = current_dataset
        else:
            rotation_dataset = torch.utils.data.ConcatDataset([rotation_dataset, current_dataset])
        del current_dataset

    # alternatively idist.utils.all_gather(torch.cat(rotation_dataset, 0).detach()) with []
    #X, Y = idist.utils.all_gather(torch.cat(rotated, 0).detach()), idist.utils.all_gather(torch.cat(angles, 0).detach())
    #rotation_dataset = torch.utils.data.TensorDataset(X, Y)

    trainloader = build_dataloader(rotation_dataset, drop_last=False)

    ## VALID to select best weights

    # rotation - use only Xs

    logger.log_msg(f"Rotating the images - valid")
    rotated, angles = [], []
    for i, batch in enumerate(valloader):
        X, Y = batch
        angle = random.choice(options)
        img = TF.rotate(X, angle)
        current_dataset = torch.utils.data.TensorDataset(torch.Tensor(img), torch.Tensor([class_to_idx[angle]] * len(img)).type(torch.LongTensor))
        if i == 0:
            rotation_dataset = current_dataset
        else:
            rotation_dataset = torch.utils.data.ConcatDataset([rotation_dataset, current_dataset])
        
    valloader = build_dataloader(rotation_dataset, drop_last=False)

    ## TEST to check against unseen

    # rotation - use only Xs

    logger.log_msg(f"Rotating the images - test")
    rotated, angles = [], []
    for i, batch in enumerate(testloader):
        X, Y = batch
        angle = random.choice(options)
        img = TF.rotate(X, angle)
        current_dataset = torch.utils.data.TensorDataset(torch.Tensor(img), torch.Tensor([class_to_idx[angle]] * len(img)).type(torch.LongTensor))
        if i == 0:
            rotation_dataset = current_dataset
        else:
            rotation_dataset = torch.utils.data.ConcatDataset([rotation_dataset, current_dataset])
        
    testloader = build_dataloader(rotation_dataset, drop_last=False)
         

    for ckpt_path in sorted(args.ckpt):
        engine_mock = get_engine_mock(ckpt_path=ckpt_path)

        logger.log_msg(f"Evaluating {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone = load_backbone(args)
        backbone.load_state_dict(ckpt['backbone'])

        build_model = partial(idist.auto_model, sync_bn=True)
        backbone   = build_model(backbone)

        # EXTRACT FROZEN FEATURES
        logger.log_msg('collecting features ...')
        X_train, Y_train = collect_features(backbone, trainloader, device, normalize=False)
        X_val,   Y_val   = collect_features(backbone, valloader,   device, normalize=False)
        X_test,  Y_test  = collect_features(backbone, testloader,  device, normalize=False)
        classifier = nn.Linear(args.num_backbone_features, num_classes).to(device)
        optim_kwargs = {
            'line_search_fn': 'strong_wolfe',
            'max_iter': 5000,
            'lr': 1.,
            'tolerance_grad': 1e-10,
            'tolerance_change': 0,
        }
        logger.log_msg('collecting features ... done')

        best_acc = 0.
        best_w = 0.
        best_classifier = None
        for w in torch.logspace(-6, 5, steps=45).tolist():
            optimizer = optim.LBFGS(classifier.parameters(), **optim_kwargs)
            optimizer.step(build_step(X_train, Y_train, classifier, optimizer, w))
            acc = compute_accuracy(X_val, Y_val, classifier)

            if best_acc < acc:
                best_acc = acc
                best_w = w
                best_classifier = deepcopy(classifier)

            logger.log_msg(f'w={w:.4e}, acc={acc:.4f}')
            logger.log(
                engine=engine_mock, global_step=-1,
                **{
                    "w": w,
                    f"predict_augmentation/lin-{args.augm}-{args.dataset}": acc
                }
            )
            if wandb.run is not None:
                wandb.log({
                    "w": w,
                    f"predict_augmentation/lin-{args.augm}-{args.dataset}": acc
                })

        logger.log_msg(f'BEST: w={best_w:.4e}, acc={best_acc:.4f}')

        X = torch.cat([X_train, X_val], 0)
        Y = torch.cat([Y_train, Y_val], 0)
        optimizer = optim.LBFGS(best_classifier.parameters(), **optim_kwargs)
        optimizer.step(build_step(X, Y, best_classifier, optimizer, best_w))
        acc = compute_accuracy(X_test, Y_test, best_classifier)
        logger.log_msg(f'test acc={acc:.4f}')
        logger.log(
            engine=engine_mock, global_step=-1,
            **{
                f"test_predict_augmentation/lin-{args.augm}-{args.dataset}": acc
            }
        )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, nargs="+")
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--augm', type=str, default='rotation')
    parser.add_argument('--logdir', type=str, default=None)
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)

