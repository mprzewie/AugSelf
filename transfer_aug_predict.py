from argparse import ArgumentParser
from collections import defaultdict
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

from cond_utils import AUG_DESC_SIZE_CONFIG
from datasets import load_datasets, load_pretrain_datasets
from models import load_backbone, load_ss_predictor
from trainers import collect_features, SSObjective
from trainers_cond import prepare_training_batch
from utils import Logger, get_engine_mock
from tqdm import tqdm




def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()


    # ckpt_parents = set([Path(c).parent for c in args.ckpt])
    # assert len(set(ckpt_parents)) == 1, f"Expected a single checkpoints directory but got {ckpt_parents}"
    logdir = Path(args.ckpt).parent

    args.origin_run_name = logdir.name
    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"aug-{args.dataset}", args=args,
        job_type="eval_aug_predict"
    )

    # DATASETS
    datasets = load_pretrain_datasets(
        dataset=args.dataset,
        datadir=args.datadir,
    )

    t1 = datasets["t1"]
    t2 = datasets["t2"]



    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=False)
    # valloader   = build_dataloader(datasets['val'],   drop_last=False)
    testloader  = build_dataloader(datasets['test_train_like'],  drop_last=False)
    # trainvalloader = build_dataloader(datasets["trainval"], drop_last=False)



    # num_classes = datasets['num_classes']


    engine_mock = get_engine_mock(ckpt_path=args.ckpt)

    logger.log_msg(f"Evaluating {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
        
        
    backbone = load_backbone(args)
    backbone.load_state_dict(ckpt['backbone'])

    build_model = partial(idist.auto_model, sync_bn=True)
    backbone   = build_model(backbone)
    backbone.eval()

    ft_dicts = dict()
    ss_objective = SSObjective(
        crop=1,
        color=1,
        flip=1,
        blur=1,
        color_diff=1,
    )
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = {k: build_model(v) for k, v in ss_predictor.items()}
    ss_opt = torch.optim.SGD(
        sum([list(v.parameters()) for v in ss_predictor.values()], []),
        lr=args.lr, weight_decay=args.wd, momentum=args.momentum
    )


    for e in range(args.epochs):
        metrics = defaultdict(list)
        for batch in tqdm(trainloader, f"{e}: train"):
            (x1, x2), (aug_d1, aug_d2), (diff1, diff2) = prepare_training_batch(batch, t1, t2, device)

            with torch.no_grad():
                y1 = backbone(x1)
                y2 = backbone(x2)

            ss_losses = ss_objective(ss_predictor, y1, y2, diff1, diff2)
            ss_losses["total"].backward()
            ss_opt.step()
            for k, v in ss_losses.items():
                metrics[f"train/{k}"].append(v.item())
            
        from pprint import pprint
        pprint({k: v.item() for (k,v) in ss_losses.items()})

        for batch in tqdm(testloader, f"{e}: test"):
            with torch.no_grad():
                (x1, x2), (aug_d1, aug_d2), (diff1, diff2) = prepare_training_batch(batch, t1, t2, device)
                y1 = backbone(x1)
                y2 = backbone(x2)
                ss_losses = ss_objective(ss_predictor, y1, y2, diff1, diff2)
                for k, v in ss_losses.items():
                    metrics[f"test/{k}"].append(v.item())

        logger.log(
            engine_mock, e,
            **{
                f"test_predict_aug/{args.dataset}/{k}": np.mean(v)
                for (k,v) in metrics.items()
            }
        )




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)

