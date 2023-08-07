from argparse import ArgumentParser
from functools import partial
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
import wandb
from tqdm import tqdm

from datasets import load_datasets
from models import load_backbone
from trainers import collect_features
from utils import Logger, get_engine_mock

import math



def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()


    logdir = args.ckpt.parent

    args.origin_run_name = logdir.name
    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"ft-{args.dataset}", args=args,
        job_type="eval_finetune_l2sp"
    )

    # DATASETS
    datasets = load_datasets(dataset=args.dataset,
                             datadir=args.datadir,
                             pretrain_data=args.pretrain_data,
                             train_crop_mode="random"
                             )
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=False)
    valloader   = build_dataloader(datasets['val'],   drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)
    trainvalloader = build_dataloader(datasets["trainval"], drop_last=False)



    engine_mock = get_engine_mock(ckpt_path=args.ckpt)

    logger.log_msg(f"Evaluating {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)


    backbone = load_backbone(args).to(device)
    backbone.load_state_dict(ckpt['backbone'])

    initial_bkb_parameters = [p.clone() for p in backbone.parameters()]

    print("Finding max class")
    y_max = -1

    loader_length = 0
    for _, y in trainloader:
        y_max = max(y.max(), y_max)
        loader_length += 1

    print("Num classes =", y_max+1)
    print("Loader length =", loader_length)
    num_epochs = math.ceil(args.l2sp_iterations / loader_length)
    print(f"Estimating number of epochs: {args.l2sp_iterations} / {loader_length} ~= {num_epochs}")


    classifier = nn.Linear(args.num_backbone_features, y_max+1)
    model = nn.Sequential(backbone, classifier).to(device)

    assert len(list(model[0].parameters())) == len(initial_bkb_parameters)
    assert len(list(model[1].parameters())) == 2

    # build_model = partial(idist.auto_model, sync_bn=True)
    # model = build_model(model)


    optimizer = optim.SGD(model.parameters(), lr=args.l2sp_lr_init, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.l2sp_lr_decrease], gamma=0.1)

    for epoch in range(num_epochs):
        rs_train = []
        rs_val = []

        model.train()
        for X, y in tqdm(trainloader, f"{epoch}: Train"):
            optimizer.zero_grad()
            y_pred = model(X.to(device))
            y_pred_cls = y_pred.argmax(dim=1)
            y = y.to(device)
            ce_loss = nn.functional.cross_entropy(y_pred, y)
            l2_loss_bkb = 0.5 * args.l2sp_alpha * sum([
                ((p_i - p_c).norm(2) ** 2)
                for (p_i, p_c)
                in zip(initial_bkb_parameters, list(model[0].parameters()))
            ])
            l2_loss_cls = 0.5 * args.l2sp_beta * sum([
                p_c.norm(2) ** 2
                for p_c in model[1].parameters()
            ])
            loss = ce_loss + l2_loss_cls + l2_loss_bkb
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc = (y_pred_cls == y).float().mean()

            r = {
                "ce": ce_loss.item(),
                "l2_bkb": l2_loss_bkb.item(),
                "l2_cls": l2_loss_cls.item(),
                "loss": loss.item(),
                "acc": acc.item(),
                "lr": scheduler.get_last_lr()
            }
            r = {
                "epoch": epoch,
                **{f"test_l2sp/{args.dataset}/{k}/train": v for (k,v) in r.items()}
            }
            rs_train.append(r)
            logger.log(engine=engine_mock, global_step=-1, **r)
        print(r)

        model.eval()

        with torch.no_grad():
            for X, y in tqdm(valloader, f"{epoch}: Val"):
                y = y.to(device)
                y_pred = model(X.to(device))
                y_pred_cls = y_pred.argmax(dim=1)
                ce_loss = nn.functional.cross_entropy(y_pred, y.to(device))
                acc = (y_pred_cls == y).float().mean()

                r = {
                    "ce": ce_loss.item(),
                    "acc": acc.item(),
                }
                r = {
                    "epoch": epoch,
                    **{f"test_l2sp/{args.dataset}/{k}/val": v for (k,v) in r.items()}
                }
                rs_val.append(r)

                logger.log(engine=engine_mock, global_step=-1, **r)

    y_test = []
    y_test_pred = []
    for X, y in tqdm(testloader, "Test"):
        y_test.extend(y.cpu().numpy())
        y_pred = model(X.to(device))
        y_pred_cls = y_pred.argmax(dim=1)
        y_test_pred.extend(y_pred_cls.detach().cpu().numpy())

    y_test = np.array(y_test)
    y_test_pred = np.array(y_test_pred)
    test_acc = (y_test == y_test_pred).mean()
    logger.log(
        engine=engine_mock, global_step=-1,
        **{
            f"test_l2sp/{args.dataset}/test": test_acc
        }
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument("--l2sp-iterations", type=int, default=9000)
    parser.add_argument("--l2sp-lr-decrease", type=int, default=6000)
    parser.add_argument("--l2sp-lr-init", type=float, default=0.02) # 0.005, 0.01, 0.02
    parser.add_argument("--l2sp-alpha", type=float, default=0.001) # {0.001, 0.01, 0.1, 1}
    parser.add_argument("--l2sp-beta", type=float, default=0.01) # {0.001, 0.01, 0.1, 1}

    args = parser.parse_args()
    args.backend = None #= 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)

