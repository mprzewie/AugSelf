import json
import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from pathlib import Path

import ignite.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from ignite.engine import Events
import ignite.distributed as idist

from cond_utils import AUG_DESC_SIZE_CONFIG, AUG_STRATEGY, AugProjector, AUG_HN_TYPES, AUG_DESC_TYPES, \
    AUG_INJECTION_TYPES
from datasets import load_pretrain_datasets
from decoders import load_decoder
from models import load_backbone, load_mlp, load_ss_predictor
import trainers_regen as trainers
from regen_utils import ReGenerator
from trainers import SSObjective
from utils import Logger, get_first_free_port
import vits



def simsiam(args, t1, t2):
    out_dim = 2048
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
        color_diff=args.ss_color_diff,
        rot=args.ss_rot,
        sol=args.ss_sol,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))

    # num_aug_features = sum(AUG_DESC_SIZE_CONFIG.values())
    sorted_aug_cond = sorted(args.aug_cond)
    n_aug_feats = sum([AUG_DESC_SIZE_CONFIG[k] for k in sorted_aug_cond])


    proj = AugProjector(
            args,
            proj_hidden_dim=out_dim,
            proj_out_dim=out_dim,
            proj_depth=2+int(args.dataset.startswith('imagenet')),
            projector_last_bn=True,
            projector_last_bn_affine=True

        )
    cond_projector = build_model(proj) if not args.no_proj else proj

    predictor    = build_model(
        load_mlp(out_dim,
        out_dim // 4,
        out_dim,
        num_layers=2,
        last_bn=False)
    )
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(cond_projector.parameters()) +ss_params),
                  build_optim(list(predictor.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.simsiam(backbone=backbone,
                               projector=cond_projector,
                               predictor=predictor,
                               ss_predictor=ss_predictor,
                               t1=t1, t2=t2,
                               optimizers=optimizers,
                               device=device,
                               ss_objective=ss_objective,
                               aug_cond=sorted_aug_cond,
                               simclr_loss = args.simsiam_use_negatives
                               )

    return dict(backbone=backbone,
                projector=cond_projector,
                predictor=predictor,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)



def moco(args, t1, t2):
    out_dim = 128
    device = idist.device()

    assert args.aug_inj_type != AUG_INJECTION_TYPES.img_cat, "Not implemented yet"

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
        color_diff=args.ss_color_diff,
        rot=args.ss_rot,
        sol=args.ss_sol,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))


    proj = AugProjector(
            args,
            proj_out_dim=out_dim,
            proj_depth=2,
        )
    projector : AugProjector = build_model(proj) if ((not args.no_proj) or (args.aug_inj_type != AUG_INJECTION_TYPES.proj_none)) else proj


    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [
        build_optim(
            list(backbone.parameters())+
            list(projector.parameters()) +
            ss_params #+
            # list(aug_bkb_projector.parameters())
        )
    ]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.moco(
            backbone=backbone,
            projector=projector,
            ss_predictor=ss_predictor,
            t1=t1, t2=t2,
            optimizers=optimizers,
            device=device,
            ss_objective=ss_objective,
            aug_cond=args.aug_cond or [],
            ifm_epsilon=args.ifm_epsilon,
            ifm_alpha=args.ifm_alpha
    )

    return dict(backbone=backbone,
                projector=projector,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)

def mocov3(
        args, t1, t2,
        # vit-b args
        stop_grad_conv1: bool=True,
        moco_dim: int=256,
        moco_mlp_dim: int=4096,
        T: float=0.2,
        warmup_epochs: int=40,


):

    # lr = 1.5e-4
    # wd = .1
    #--optimizer = adamw - -lr = 1.5e-4 - -weight - decay = .1 \
    # --epochs = 300 - -warmup - epochs = 40 \
    # --stop - grad - conv1 - -moco - m - cos - -moco - t = .2 \
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
        color_diff=args.ss_color_diff,
        rot=args.ss_rot,
        sol=args.ss_sol,
    )

    sorted_aug_cond = sorted(args.aug_cond or [])

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone = build_model(load_backbone(args))
    

    projector: AugProjector= build_model(
        AugProjector(
            args,
            proj_hidden_dim=moco_mlp_dim,
            proj_out_dim=moco_dim,
            proj_depth=3,
            projector_last_bn=True, projector_last_bn_affine=False
        )
    )

    predictor =  build_model(
        load_mlp(moco_dim,
        moco_mlp_dim,
        moco_dim,
        num_layers=2,
        last_bn=True, last_bn_affine=False)
    )

    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    AdamW = partial(optim.AdamW, lr=args.lr, weight_decay=args.wd)
    build_optim = lambda x: idist.auto_optim(AdamW(x))
    optimizers = [
        build_optim(list(backbone.parameters())+list(projector.parameters()) + list(predictor.parameters())+ss_params)
    ]
    schedulers = [
        optim.lr_scheduler.SequentialLR(
            optimizers[0],
            [
                optim.lr_scheduler.LinearLR(
                    optimizers[0],
                    start_factor=1 / warmup_epochs,
                    end_factor=1,
                    total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], args.max_epochs - warmup_epochs
                )
            ],
            milestones=[warmup_epochs]
        )
    ]
    trainer = trainers_cond.mocov3(
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        ss_predictor=ss_predictor,
        t1=t1,t2=t2,
        optimizers=optimizers,
        device=device,
        ss_objective=ss_objective,
        aug_cond=sorted_aug_cond,
    )

    return dict(
        backbone=backbone,
        projector=projector,
        ss_predictor=ss_predictor,
        optimizers=optimizers,
        schedulers=schedulers,
        trainer=trainer
    )




def simclr(args, t1, t2):
    out_dim = 128
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
        color_diff=args.ss_color_diff,
        rot=args.ss_rot,
        sol=args.ss_sol,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))

    sorted_aug_cond = sorted(args.aug_cond)

    proj = AugProjector(
            args,
            proj_out_dim=out_dim,
            proj_depth=2,
        )
    projector = build_model(proj) if ((not args.no_proj) or (args.aug_inj_type != AUG_INJECTION_TYPES.proj_none)) else proj


    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters())+ss_params)]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.simclr(
        backbone=backbone,
        projector=projector,
        ss_predictor=ss_predictor,
        t1=t1, t2=t2,
        optimizers=optimizers,
        device=device,
        ss_objective=ss_objective,
        aug_cond=sorted_aug_cond,
    )

    return dict(backbone=backbone,
                projector=projector,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)

def barlow_twins(
        args, t,
        out_dim: int=8192,
        warmup_epochs: int=10,
        lr_bias_scale: float = 0.024
):
    device = idist.device()


    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = load_backbone(args)
    decoder = load_decoder(args)
    regenerator = build_model(ReGenerator(backbone, decoder))

    projector = build_model(load_mlp(
             args.num_backbone_features,
             out_dim,
             out_dim,
             num_layers=3,
             last_bn=True,
            last_bn_affine=False
    ))
    projector_copy = deepcopy(projector)

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    build_optim = lambda x: idist.auto_optim(SGD(x))
    parameters = list(backbone.parameters())+ list(decoder.parameters()) +  list(projector.parameters())
    param_weights = [p for p in parameters if p.ndim != 1]
    param_biases = [p for p in parameters if p.ndim == 1]
    optimizers = [
        build_optim([
            {
                'params': param_weights,
                "lr": args.lr,
            },
            {
                'params': param_biases,
                "lr": args.lr * lr_bias_scale
             }
        ])
    ]
    schedulers = [
        optim.lr_scheduler.SequentialLR(
            optimizers[0],
            [
                optim.lr_scheduler.LinearLR(
                    optimizers[0],
                    start_factor=1 / warmup_epochs,
                    end_factor=1,
                    total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], args.max_epochs - warmup_epochs
                )
            ],
            milestones=[warmup_epochs]
        )
    ]

    trainer = trainers.barlow_twins(
        regenerator=regenerator,
        projector=projector,
        projector_copy=projector_copy,
        optimizers=optimizers,
        device=device,
        t=t,
        batch_size = args.batch_size
    )

    return dict(backbone=backbone,
                projector=projector,
                decoder=decoder,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer,
                )

def byol(args, t1, t2):
    out_dim = 256
    h_dim = 4096
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
        color_diff=args.ss_color_diff,
        rot=args.ss_rot,
        sol=args.ss_sol,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        h_dim,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    predictor    = build_model(load_mlp(out_dim,
                                        h_dim,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters())+ss_params+list(predictor.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.byol(backbone=backbone,
                            projector=projector,
                            predictor=predictor,
                            ss_predictor=ss_predictor,
                            t1=t1, t2=t2,
                            optimizers=optimizers,
                            device=device,
                            ss_objective=ss_objective)

    return dict(backbone=backbone,
                projector=projector,
                predictor=predictor,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def swav(args, t1, t2):
    out_dim = 128
    h_dim = 2048
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
        color_diff=args.ss_color_diff,
        rot=args.ss_rot,
        sol=args.ss_sol,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))

    sorted_aug_cond = sorted(args.aug_cond)

    cond_projector = build_model(
        AugProjector(
            args,
            proj_hidden_dim=h_dim,
            proj_out_dim=out_dim,
            proj_depth=2,
            projector_last_bn=False,
        )
    )

    prototypes   = build_model(nn.Linear(out_dim, 100, bias=False))
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(cond_projector.parameters())+ss_params+list(prototypes.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.swav(backbone=backbone,
                            projector=cond_projector,
                            prototypes=prototypes,
                            ss_predictor=ss_predictor,
                            t1=t1, t2=t2,
                            optimizers=optimizers,
                            device=device,
                            ss_objective=ss_objective,
                            aug_cond=sorted_aug_cond
                            )

    return dict(backbone=backbone,
                projector=cond_projector,
                prototypes=prototypes,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()

    logger = Logger(
        args.logdir, args.resume, args=args,
        job_type="pretrain_regen"
    )

    if idist.get_rank() == 0:
        with (Path(args.logdir) / "rerun.sh").open("w") as f:
            print("python", " ".join(sys.argv), file=f)

        with (Path(args.logdir) / "args.json").open("w") as f:
            json.dump(
                {
                    k: v if isinstance(v, (int, str, bool, float)) else str(v)
                    for (k, v) in vars(args).items()
                },
                f,
                indent=2,
            )

    # DATASETS
    logger.log_msg(f"{args.seed=}")
    logger.log_msg(f"Loading {args.dataset}")
    datasets = load_pretrain_datasets(dataset=args.dataset,
                                      datadir=args.datadir,
                                      color_aug=args.color_aug,
                                      num_views=1
                                      )

    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=True)
    valloader   = build_dataloader(datasets['val']  , drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)

    t1, t2 = datasets['t1'], datasets['t2']


    # MODELS

    logger.log_msg(f"Building {args.framework}")



    if args.framework == 'simsiam':
        models = simsiam(args, t1, t2)
    elif args.framework == 'moco':
        models = moco(args, t1, t2)
    elif args.framework == 'simclr':
        models = simclr(args, t1, t2)
    elif args.framework == "barlow_twins":
        models = barlow_twins(args, t1)
    elif args.framework == 'byol':
        models = byol(args, t1, t2)
    elif args.framework == 'swav':
        models = swav(args, t1, t2)
    elif args.framework == "mocov3":
        models = mocov3(args, t1, t2)

    trainer   = models['trainer']
    evaluator = trainers.nn_evaluator(backbone=models['backbone'],
                                      trainloader=valloader,
                                      testloader=testloader,
                                      device=device)

    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            for loader in [trainloader, valloader, testloader]:
                loader.sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.ITERATION_STARTED)
    def log_lr(engine):
        lrs = {}
        for i, optimizer in enumerate(models['optimizers']):
            for j, pg in enumerate(optimizer.param_groups):
                lrs[f'lr/{i}-{j}'] = pg['lr']
        logger.log(engine, engine.state.iteration, print_msg=False, **lrs)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log(engine):
        loss = engine.state.output.pop('loss')
        # ss_loss = engine.state.output.pop('ss/total')
        logger.log(engine, engine.state.iteration,
                   print_msg=engine.state.iteration % args.print_freq == 0,
                   loss=loss,
                   # ss_loss=ss_loss
                   )

        if 'z1' in engine.state.output:
            with torch.no_grad():
                z1 = engine.state.output.pop('z1')
                z2 = engine.state.output.pop('z2')
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                dist = torch.einsum('ik, jk -> ij', z1, z2)
                diag_masks = torch.diag(torch.ones(z1.shape[0])).bool()
                engine.state.output['dist/intra'] = dist[diag_masks].mean().item()
                engine.state.output['dist/inter'] = dist[~diag_masks].mean().item()

        logger.log(engine, engine.state.iteration,
                   print_msg=False,
                   **engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_freq))
    def evaluate(engine):
        acc = evaluator()
        logger.log(engine, engine.state.epoch, acc=acc)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        for scheduler in models['schedulers']:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED(every=args.ckpt_freq))
    def save_ckpt(engine):
        logger.save(engine, **models)
    
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def save_last_ckpt(engine):
        logger.save(engine, override_name="ckpt-last.pth", **models)

    if args.resume is not None:
        @trainer.on(Events.STARTED)
        def load_state(engine):
            resume_path = Path(args.logdir) / f'ckpt-{args.resume}.pth'
            assert resume_path.exists(), resume_path

            ckpt = torch.load(resume_path, map_location='cpu')
            for k, v in models.items():
                try:
                    if isinstance(v, nn.parallel.DistributedDataParallel):
                        v = v.module

                    if hasattr(v, 'state_dict'):
                        v.load_state_dict(ckpt[k])

                    if type(v) is list and hasattr(v[0], 'state_dict'):
                        for i, x in enumerate(v):
                            x.load_state_dict(ckpt[k][i])

                    if type(v) is dict and k == 'ss_predictor':
                        for y, x in v.items():
                            x.load_state_dict(ckpt[k][y])

                    logger.log_msg(f"Successfully loaded {k}")
                except Exception as e:
                    logger.log_msg(f"Error loading {k}: {e}")
                    if k == "backbone":
                        raise

    trainer.run(trainloader, max_epochs=args.max_epochs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet50", "vit_base", "vit_small"])
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--framework', type=str, default='barlow_twins',
                        choices=["barlow_twins"]
                        # choices=["moco", "simsiam", "simclr", "barlow_twins", "mocov3", "swav"]
                        )

    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--ckpt-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=1)

    parser.add_argument('--color-aug', type=str, default='default')


    parser.add_argument(
        "--no-proj", action="store_true", help="If true, projector becomes an identity (like in MoCo-v1)"
    )
    
    parser.add_argument(
        "--bkb-feat-dim", type=int, default=None, help="Use first N features from backbone for projector and omit the rest (like in DirectCLR). Can only be used in conjunction with --no-proj."
    )


    parser.add_argument(
        "--simsiam-use-negatives", action="store_true", default=False,
        help="Simsiam with simclr loss"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Manual seed"
    )

    parser.add_argument("--dec-mlp-depth", type=int, default=3)
    parser.add_argument("--dec-hierarchical", action="store_true", default=False)
    parser.add_argument(
        "--dec-width", type=int, default=1
    )


    parser.add_argument("--ifm-alpha", type=float, default=0.0)
    parser.add_argument("--ifm-epsilon", type=float, default=0.1)


    args = parser.parse_args()
    args.lr = args.base_lr * args.batch_size / 256

    if args.seed is not None:
        ignite.utils.manual_seed(args.seed)

    if not args.distributed:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        free_port = get_first_free_port()
        with idist.Parallel(
                'nccl',
                nproc_per_node=torch.cuda.device_count(),
                master_port=free_port
                # init_method=f"tcp://0.0.0.0:{free_port}"
        ) as parallel:
            parallel.run(main, args)

