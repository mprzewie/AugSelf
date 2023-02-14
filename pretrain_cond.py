import json
import os
import sys
from argparse import ArgumentParser
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
from models import load_backbone, load_mlp, load_ss_predictor
import trainers_cond as trainers
from trainers import SSObjective
from utils import Logger

"""
('crop',  crop,  4, 'regression'),
        ('color', color, 4, 'regression'),
        ('flip',  flip,  1, 'binary_classification'),
        ('blur',  blur,  1, 'regression'),
        ('rot',    rot,  4, 'classification'),
        ('sol',    sol,  1, 'regression'),"""


def simsiam(args, t1, t2):
    out_dim = 2048
    device  = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        rot   = args.ss_rot,
        sol   = args.ss_sol,
        only  = args.ss_only,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))

    sorted_aug_cond = sorted(args.aug_cond)
    n_aug_feats  = sum([AUG_DESC_SIZE_CONFIG[k] for k in sorted_aug_cond])
    """
    cond_projector    = build_model(
        load_mlp(
            args.num_backbone_features + n_aug_feats,
            out_dim,
            out_dim,
            num_layers=2+int(args.dataset.startswith('imagenet')),
            last_bn=True
        )
    )
    """
    cond_projector = build_model(
        AugProjector(
            args,
            proj_out_dim=out_dim,
            proj_depth=2,
        )
    )
    
    predictor    = build_model(load_mlp(out_dim,
                                        out_dim // 4,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))

    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(cond_projector.parameters())+ss_params),
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
                               aug_cond=sorted_aug_cond
                               )

    return dict(backbone=backbone,
                projector=cond_projector,
                predictor=predictor,
                # ss_predictor=ss_predictor,
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
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))


    projector : AugProjector= build_model(
        AugProjector(
            args,
            proj_out_dim=out_dim,
            proj_depth=2,
        )
    )

    # if args.aug_cnt_lambda > 0 and args.aug_treatment == AUG_STRATEGY.mlp:
    #     aug_bkb_projector = build_model(
    #         load_mlp(
    #             n_in=(
    #                 args.num_backbone_features
    #                 if args.aug_desc_type == AUG_DESC_TYPES.absolute
    #                 else 2 * args.num_backbone_features
    #             ),
    #             n_hidden=args.aug_nn_width,
    #             n_out=args.aug_processor_out,
    #             num_layers=args.aug_nn_depth,
    #         )
    #     )
    # else:
    #     aug_bkb_projector = nn.Identity()

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
            # aug_desc_type=args.aug_desc_type,
            # aug_contrastive_loss_lambda=args.aug_cnt_lambda,
            # aug_bkb_projector=aug_bkb_projector,
            # aug_treatment=args.aug_treatment,
            aug_cond=args.aug_cond or []
    )

    return dict(backbone=backbone,
                projector=projector,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)

def simclr(args, t1, t2):
    out_dim = 128
    device = idist.device()

    ss_objective = SSObjective(
        crop  = args.ss_crop,
        color = args.ss_color,
        flip  = args.ss_flip,
        blur  = args.ss_blur,
        only  = args.ss_only,
    )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    cond_projector : AugProjector= build_model(
        AugProjector(
            args,
            proj_out_dim=out_dim,
            proj_depth=2,
        )
    )
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(cond_projector.parameters())+ss_params)]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.simclr(backbone=backbone,
                              projector=cond_projector,
                              ss_predictor=ss_predictor,
                              t1=t1, t2=t2,
                              optimizers=optimizers,
                              device=device,
                              ss_objective=ss_objective,
                              aug_cond=args.aug_cond or [])

    return dict(backbone=backbone,
                projector=cond_projector,
                ss_predictor=ss_predictor,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def byol(args, t1, t2):
    out_dim = 256
    h_dim = 4096
    device = idist.device()

    # ss_objective = SSObjective(
    #     crop  = args.ss_crop,
    #     color = args.ss_color,
    #     flip  = args.ss_flip,
    #     blur  = args.ss_blur,
    #     rot   = args.ss_rot,
    #     only  = args.ss_only,
    # )

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

    # ss_objective = SSObjective(
    #     crop  = args.ss_crop,
    #     color = args.ss_color,
    #     flip  = args.ss_flip,
    #     blur  = args.ss_blur,
    #     rot   = args.ss_rot,
    #     only  = args.ss_only,
    # )

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projector    = build_model(load_mlp(args.num_backbone_features,
                                        h_dim,
                                        out_dim,
                                        num_layers=2,
                                        last_bn=False))
    prototypes   = build_model(nn.Linear(out_dim, 100, bias=False))
    ss_predictor = load_ss_predictor(args.num_backbone_features, ss_objective)
    ss_predictor = { k: build_model(v) for k, v in ss_predictor.items() }
    ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+list(projector.parameters())+ss_params+list(prototypes.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.swav(backbone=backbone,
                            projector=projector,
                            prototypes=prototypes,
                            ss_predictor=ss_predictor,
                            t1=t1, t2=t2,
                            optimizers=optimizers,
                            device=device,
                            ss_objective=ss_objective)

    return dict(backbone=backbone,
                projector=projector,
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
        job_type="pretrain"
    )

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
                                      color_aug=args.color_aug)

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


    assert args.framework in ["moco", "simsiam", "simclr"] # TODO

    if args.framework == 'simsiam':
        models = simsiam(args, t1, t2)
    elif args.framework == 'moco':
        models = moco(args, t1, t2)
    elif args.framework == 'simclr':
        models = simclr(args, t1, t2)
    elif args.framework == 'byol':
        models = byol(args, t1, t2)
    elif args.framework == 'swav':
        models = swav(args, t1, t2)

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

    if args.resume is not None:
        @trainer.on(Events.STARTED)
        def load_state(engine):
            ckpt = torch.load(os.path.join(args.logdir, f'ckpt-{args.resume}.pth'), map_location='cpu')
            for k, v in models.items():
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

    trainer.run(trainloader, max_epochs=args.max_epochs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--framework', type=str, default='simsiam')

    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--ckpt-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=1)

    parser.add_argument('--color-aug', type=str, default='default')

    parser.add_argument(
        '--aug-treatment', # TODO: aug-strategy
        type=str, default=AUG_STRATEGY.raw,
        choices=[AUG_STRATEGY.raw, AUG_STRATEGY.mlp, AUG_STRATEGY.hn]
    )
    parser.add_argument(
        "--aug-nn-width", type=int, default=32,
        help="Hidden size of aug processing network / aug hypernetwork, depending on aug-treatment"
    )
    parser.add_argument(
        "--aug-nn-depth", type=int, default=2,
        help="Depth of aug processing network / aug hypernetwork, depending on aug-treatment"
    )

    parser.add_argument(
        "--aug-hn-type", type=str, default=AUG_HN_TYPES.mlp,
        choices=[AUG_HN_TYPES.mlp, AUG_HN_TYPES.mlp_bn],
        help="Type of aug hypernetwork. Used only if aug-treatment==hn"
    )

    parser.add_argument(
        "--aug-inj-type", type=str, default=AUG_INJECTION_TYPES.proj_cat,
        choices=[
            AUG_INJECTION_TYPES.proj_cat,
            AUG_INJECTION_TYPES.proj_add,
            AUG_INJECTION_TYPES.proj_mul,
            AUG_INJECTION_TYPES.img_cat,
            AUG_INJECTION_TYPES.proj_none,

        ],
        help="How to inject raw or mlp-processed aug vectors. Used only if aug-treatment==mlp and in some cases for raw."
    )

    parser.add_argument(
        "--aug-cond", nargs="*", type=str, choices=list(AUG_DESC_SIZE_CONFIG.keys()),
        help="Augmentations to condition the projector on"
    )

    parser.add_argument(
        "--aug-desc-type", type=str, choices=[AUG_DESC_TYPES.absolute, AUG_DESC_TYPES.relative],
        default=AUG_DESC_TYPES.absolute,
        help=f"Whether to condition on absolute augmentation descriptors ({AUG_DESC_TYPES.absolute}) "
             f"or differences between them ({AUG_DESC_TYPES.relative})"
    )

    parser.add_argument(
        "--aug-cnt-lambda", type=float, default=0,
        help="Weight of backbone/augmentation contrastive loss"
    )

    parser.add_argument(
        "--seed", type=int, default=None,
        help="Manual seed"
    )
    parser.add_argument('--ss-crop',  type=float, default=-1)
    parser.add_argument('--ss-color', type=float, default=-1)
    parser.add_argument('--ss-flip',  type=float, default=-1)
    parser.add_argument('--ss-blur',  type=float, default=-1)
    parser.add_argument('--ss-rot',   type=float, default=-1)
    parser.add_argument('--ss-sol',   type=float, default=-1)
    parser.add_argument('--ss-only',  action='store_true')

    args = parser.parse_args()
    args.lr = args.base_lr * args.batch_size / 256

    if args.seed is not None:
        ignite.utils.manual_seed(args.seed)

    if not args.distributed:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        with idist.Parallel('nccl', nproc_per_node=torch.cuda.device_count()) as parallel:
            parallel.run(main, args)

