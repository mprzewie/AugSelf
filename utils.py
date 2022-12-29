import logging
import os
from pathlib import Path

import ignite.distributed as idist
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def maybe_setup_wandb(logdir, args=None):

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if wandb_entity is None or wandb_project is None:
        print(f"{wandb_entity=}", f"{wandb_project=}")
        print("Not initializing WANDB")
        return

    run_name = Path(logdir).name

    run_id = None
    api = wandb.Api()

    retrieved_runs = api.runs(f'{wandb_entity}/{wandb_project}', filters={'display_name': run_name})
    print(f'Retrieved {len(retrieved_runs)} for run_name: {run_name}')

    assert len(retrieved_runs) <= 1, f'retrieved_runs: {len(retrieved_runs)}'
    if len(retrieved_runs) == 1:
        run_id = retrieved_runs[0].id

    wandb.init(
        entity=wandb_entity, project=wandb_project, config=args, name=run_name, dir=logdir, sync_tensorboard=True,
        id=run_id, resume="allow"
    )

    print("WANDB run", wandb.run.id, run_name)

def get_engine_mock(ckpt_path: str):
    try:
        epoch_no = int(
            ckpt_path.replace(".pth", "").replace("ckpt-", "")
        )
    except:
        epoch_no = -1

    class engine:
        class state:
            epoch = epoch_no
            iteration = epoch_no

    return engine


class Logger(object):

    def __init__(self, logdir, resume=None, args=None):
        assert logdir is not None

        self.logdir = logdir
        self.rank = idist.get_rank()

        handlers = [logging.StreamHandler(os.sys.stdout)]
        if logdir is not None and self.rank == 0:
            if resume is None:
                os.makedirs(logdir)
            maybe_setup_wandb(logdir=logdir, args=args)
            handlers.append(logging.FileHandler(os.path.join(logdir, 'log.txt')))
            self.writer = SummaryWriter(log_dir=logdir)
        else:
            self.writer = None

        logging.basicConfig(format=f"[%(asctime)s ({self.rank})] %(message)s",
                            level=logging.INFO,
                            handlers=handlers)
        logging.info(' '.join(os.sys.argv))

    def log_msg(self, msg):
        if idist.get_rank() > 0:
            return
        logging.info(msg)

    def log(self, engine, global_step, print_msg=True, **kwargs):
        msg = f'[epoch {engine.state.epoch}] [iter {engine.state.iteration}]'
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is float:
                msg += f' [{k} {v:.4f}]'
            else:
                msg += f' [{k} {v}]'

            if self.writer is not None:
                self.writer.add_scalar(k, v, global_step)

        if print_msg:
            logging.info(msg)

    def save(self, engine, **kwargs):
        if idist.get_rank() > 0:
            return

        state = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                v = v.module

            if hasattr(v, 'state_dict'):
                state[k] = v.state_dict()

            if type(v) is list and hasattr(v[0], 'state_dict'):
                state[k] = [x.state_dict() for x in v]

            if type(v) is dict and k == 'ss_predictor':
                state[k] = { y: x.state_dict() for y, x in v.items() }

        torch.save(state, os.path.join(self.logdir, f'ckpt-{engine.state.epoch}.pth'))

