from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import ignite.distributed as idist
import numpy as np
import sklearn
import torch
import torch.backends.cudnn as cudnn
import wandb
from sklearn.neighbors import NearestNeighbors

from datasets import load_datasets_for_cosine_sim
from resnets import load_backbone_out_blocks
from utils import Logger, get_engine_mock


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()

    logdir = Path(args.ckpt).parent

    args.origin_run_name = logdir.name

    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"feat_nn-{args.dataset}-{args.nn_metric}", args=args,
        job_type="eval_nearest_neighbors"

    )

    # DATASETS

    datasets = load_datasets_for_cosine_sim(
        dataset=args.dataset,
        pretrain_data=args.pretrain_data,
        datadir=args.datadir,
    )

    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False,
                               pin_memory=True)

    testloader  = build_dataloader(datasets['test'],  drop_last=False)
    dataset_test_no_transforms = datasets["test_no_transform"]

    transforms_dict = datasets["transforms"]

    ckpt_path = args.ckpt
    engine_mock = get_engine_mock(ckpt_path=ckpt_path)

    logger.log_msg(f"Evaluating {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = load_backbone_out_blocks(args)


    backbone.load_state_dict(ckpt['backbone'])

    build_model = partial(idist.auto_model, sync_bn=True)
    backbone = build_model(backbone)

    # EXTRACT FROZEN FEATURES
    logger.log_msg('collecting features ...')

    latents = []

    with torch.no_grad():
        for i, (X, _) in enumerate(testloader):
            X_transformed = {
                t_name: t(X) for (t_name, t) in transforms_dict.items()
            }
            X_norm = X_transformed.pop("identity")

            bs = X_norm.shape[0]

            feats_norm = backbone(X_norm.to(device))["backbone_out"].detach().cpu().numpy()

            latents.append(feats_norm)

    latents = np.concatenate(latents)

    nn = NearestNeighbors(
        n_neighbors=args.n_neighbors,
        metric=args.nn_metric
    ).fit(latents)

    query_latents = latents[:args.n_queries]

    query_nns = nn.kneighbors(
        query_latents, n_neighbors=args.n_neighbors, return_distance=False
    )

    for i, nn_indices in enumerate(query_nns):
        query_img = dataset_test_no_transforms[i]
        result_imgs = [dataset_test_no_transforms[n] for n in nn_indices]
        wandb.log(
            {
                f"query_{i}": [wandb.Image(query_img)],
                f"results_{i}": [wandb.Image(r) for r in result_imgs]
            }
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--nn-metric", type=str, default="cosine", choices=sklearn.metrics.pairwise.distance_metrics())
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)
