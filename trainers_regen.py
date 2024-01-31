import math
from copy import deepcopy
from typing import Dict, Tuple, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine
import ignite.distributed as idist

from cond_utils import AugProjector, AUG_DESC_TYPES, AUG_STRATEGY
from models import load_mlp
from regen_utils import ReGenerator
from trainers import SSObjective
from transforms import extract_aug_descriptors, extract_diff


def prepare_training_batch(batch, transforms, device) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
]:
    ((x, w),), _ = batch  # TODO potentially expand to multiple views
    return transforms(x.to(device)).detach()


def simsiam(backbone,
            projector: AugProjector,
            predictor: nn.Module,
            ss_predictor: Dict[str, nn.Module],
            t1,
            t2,
            optimizers,
            device,
            ss_objective: SSObjective,
            aug_cond,
            simclr_loss: bool = False
            ):
    raise NotImplementedError()

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        predictor.train()

        for o in optimizers:
            o.zero_grad()

        (x1, x2), (desc1, desc2), (diff1, diff2) = prepare_training_batch(batch, t1, t2, device)
        y1, y2 = backbone(x1), backbone(x2)

        aug_ks = sorted(aug_cond)
        d1_cat = torch.cat([desc1[k] for k in aug_ks], dim=1)
        d2_cat = torch.cat([desc2[k] for k in aug_ks], dim=1)

        if True:  # not ss_objective.only:
            z1 = projector(y1, d1_cat)
            z2 = projector(y2, d2_cat)
            p1 = predictor(z1)
            p2 = predictor(z2)

            if not simclr_loss:
                loss1 = F.cosine_similarity(p1, z2.detach(), dim=-1).mean().mul(-1)
                loss2 = F.cosine_similarity(p2, z1.detach(), dim=-1).mean().mul(-1)
                loss = (loss1 + loss2).mul(0.5)
            else:
                T = 0.2
                z = torch.cat([z1, z2], 0)
                scores = torch.einsum('ik, jk -> ij', z, z).div(T)
                n = z1.shape[0]
                labels = torch.tensor(list(range(n, 2 * n)) + list(range(0, n)), device=scores.device)
                masks = torch.zeros_like(scores, dtype=torch.bool)
                for i in range(2 * n):
                    masks[i, i] = True
                scores = scores.masked_fill(masks, float('-inf'))
                loss = F.cross_entropy(scores, labels)

        else:
            loss = 0.

        total_loss = (regen_lambda * loss) + (ae_lambda * ae_loss)
        outputs = dict(loss=total_loss, z1=z1, z2=z2, regen_loss=loss, ae_loss=ae_loss)

        if not ss_objective.only:
            outputs['z1'] = z1
            outputs['z2'] = z2

        ss_losses = ss_objective(ss_predictor, y1, y2, diff1, diff2)
        (loss + ss_losses['total']).backward()

        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def moco(backbone,
         projector: AugProjector,
         ss_predictor: Dict[str, nn.Module],
         t1,
         t2,
         optimizers,
         device,
         ss_objective: SSObjective,
         aug_cond: List[str],
         momentum=0.999,
         K: int = 65536,
         T: float = 0.2,
         ifm_epsilon: float = 0.0,
         ifm_alpha: float = 0.1
         ):
    target_backbone = deepcopy(backbone)
    target_projector = deepcopy(projector)
    # target_aug_bkb_projector = deepcopy(aug_bkb_projector)
    for p in (
            list(target_backbone.parameters()) +
            list(target_projector.parameters())
            # list(aug_bkb_projector.parameters())
    ):
        p.requires_grad = False

    _proj = projector if isinstance(projector, AugProjector) else projector.module
    if _proj.no_proj:
        queue = F.normalize(torch.randn(K, 2048).to(device)).detach()
    else:
        queue = F.normalize(torch.randn(K, 128).to(device)).detach()

    queue.requires_grad = False
    queue.ptr = 0

    def training_step(engine, batch):
        backbone.train()
        projector.train()

        target_backbone.train()
        target_projector.train()

        for o in optimizers:
            o.zero_grad()

        (x1, x2), (aug_d1, aug_d2), (diff1, diff2) = prepare_training_batch(batch, t1, t2, device)

        aug_keys = sorted(aug_cond)

        d1_cat = torch.concat([aug_d1[k] for k in aug_keys], dim=1)
        d2_cat = torch.concat([aug_d2[k] for k in aug_keys], dim=1)

        y1 = backbone(x1)
        z1 = F.normalize(
            projector(y1, d1_cat)
        )

        with torch.no_grad():
            y2 = target_backbone(x2)
            z2 = F.normalize(
                target_projector(y2, d2_cat)
            )

        l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [z1, queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1).div(T)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, labels)

        if ifm_epsilon > 0:
            logits_adv = torch.cat([l_pos - ifm_epsilon, l_neg + ifm_epsilon], dim=1).div(T)
            loss_adv = F.cross_entropy(logits_adv, labels)
            loss = loss + ifm_alpha * loss_adv
            loss = loss / (1 + ifm_alpha)

        outputs = dict(loss=loss, z1=z1, z2=z2)

        ss_losses = ss_objective(ss_predictor, y1, y2, diff1, diff2)

        (loss + ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        # momentum network update
        for online, target in [
            (backbone, target_backbone), (projector, target_projector)
        ]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data.mul_(momentum).add_(p1.data, alpha=1 - momentum)

        # queue update
        keys = idist.utils.all_gather(z1)

        queue[queue.ptr:queue.ptr + keys.shape[0]] = keys
        queue.ptr = (queue.ptr + keys.shape[0]) % K

        return outputs

    engine = Engine(training_step)
    return engine


def mocov3(
        backbone,
        projector: AugProjector,
        predictor: nn.Module,
        ss_predictor: Dict[str, nn.Module],
        t1,
        t2,
        optimizers,
        device,
        ss_objective: SSObjective,
        aug_cond: List[str],
        momentum=0.999,
        T: float = 0.2, ):
    target_backbone = deepcopy(backbone)
    target_projector = deepcopy(projector)

    for p in (
            list(target_backbone.parameters()) +
            list(target_projector.parameters())
    ):
        p.requires_grad = False

    def mv3_contrastive_loss(q, k):
        k = idist.all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * idist.get_rank()).to(device)
        return F.cross_entropy(logits, labels) * (2 * T)

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        predictor.train()

        target_backbone.train()
        target_projector.train()

        for o in optimizers:
            o.zero_grad()

        (x1, x2), (aug_d1, aug_d2), (diff1, diff2) = prepare_training_batch(batch, t1, t2, device)

        aug_keys = sorted(aug_cond)

        d1_cat = torch.concat([aug_d1[k] for k in aug_keys], dim=1)
        d2_cat = torch.concat([aug_d2[k] for k in aug_keys], dim=1)

        y1 = backbone(x1)
        y2 = backbone(x2)

        q1 = F.normalize(predictor(projector(y1, d1_cat)), dim=1)
        q2 = F.normalize(predictor(projector(y2, d2_cat)), dim=1)

        with torch.no_grad():
            for online, target in [
                (backbone, target_backbone), (projector, target_projector)
            ]:
                for p1, p2 in zip(online.parameters(), target.parameters()):
                    p2.data.mul_(momentum).add_(p1.data, alpha=1 - momentum)

            k1 = F.normalize(target_projector(target_backbone(x1), d1_cat), dim=1)
            k2 = F.normalize(target_projector(target_backbone(x2), d2_cat), dim=1)

        loss = mv3_contrastive_loss(q1, k2) + mv3_contrastive_loss(q2, k1)

        outputs = dict(loss=loss, z1=torch.concat([q1, q2], dim=0), z2=torch.concat([k2, k1], dim=0))
        ss_losses = ss_objective(ss_predictor, y1, y2, diff1, diff2)

        (loss + ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        return outputs

    engine = Engine(training_step)
    return engine


def simclr(regenerator: ReGenerator,
           projector: nn.ModuleDict,
           projector_copy: nn.ModuleDict,
           t,
           optimizers,
           device,
           regen_lambda: float,
           ae_lambda: float,
           inputs_to_projector: List[str],
           T: float = 0.2,
           ):
    def training_step(engine, batch):
        regenerator.train()
        projector.train()
        projector_copy.train()

        # from time import time
        for o in optimizers:
            o.zero_grad()

        # s = time()
        X = prepare_training_batch(batch, transforms=t, device=device)
        # t1 = time()

        true_embedding, regen_embedding, regen_X, ae_loss = regenerator(X, reset_backbone_copy=True)

        # t2 = time()
        projector_copy.load_state_dict(projector.state_dict())
        projector_copy.zero_grad()

        zs = dict()

        ssl_losses = dict()
        total_ssl_loss = 0

        for layer_id in inputs_to_projector:
            e_true = true_embedding[layer_id].mean(dim=(2,3))
            e_regen = regen_embedding[layer_id].mean(dim=(2,3))

            z1 = F.normalize(projector[layer_id](e_true))
            z2 = F.normalize(projector_copy[layer_id](e_regen))
            z = torch.cat([z1, z2], 0)
            scores = torch.einsum('ik, jk -> ij', z, z).div(T)
            n = z1.shape[0]
            labels = torch.tensor(list(range(n, 2 * n)) + list(range(0, n)), device=scores.device)
            masks = torch.zeros_like(scores, dtype=torch.bool)
            for i in range(2 * n):
                masks[i, i] = True
            scores = scores.masked_fill(masks, float('-inf'))
            loss = F.cross_entropy(scores, labels)
            ssl_losses[f"ssl_loss/{layer_id}"] = loss
            total_ssl_loss = total_ssl_loss + loss

            if layer_id == "l4":
                zs["z1"] = z1
                zs["z2"] = z2

        ssl_losses["ssl_loss/total"] = total_ssl_loss
        total_loss = (regen_lambda * total_ssl_loss) + (ae_lambda * ae_loss)

        outputs = dict(loss=total_loss,  ae_loss=ae_loss, **ssl_losses, **zs)
        # t5 = time()

        total_loss.backward()

        # assert False, list(regenerator.decoder.parameters())[0].grad

        # t6 = time()
        for o in optimizers:
            o.step()

        # t7 = time()
        #
        # times = dict(
        #     batch_prepare=(t1-s),
        #     regenerator=(t2-t1),
        #     projector_copy_prepare=(t3-t2),
        #     projectors=(t4-t3),
        #     loss_calc=(t5-t4),
        #     loss_back=(t6-t5),
        #     opt_step=(t7-t6),
        #     sum=(t7 - s)
        # )

        # from pprint import pprint
        # pprint(times)

        return outputs

    engine = Engine(training_step)
    return engine


def barlow_twins(
        regenerator: ReGenerator,
        projector: nn.Module,
        projector_copy: nn.Module,
        t,
        optimizers,
        device,
        batch_size: int,
        regen_lambda: float,
        ae_lambda,
        bt_lambda: float = 0.0051,
):
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def training_step(engine, batch):
        regenerator.train()
        projector.train()
        projector_copy.train()

        for o in optimizers:
            o.zero_grad()

        X = prepare_training_batch(batch, transforms=t, device=device)

        true_embedding, regen_embedding, regen_X, ae_loss = regenerator(X, reset_backbone_copy=True)
        projector_copy.load_state_dict(projector.state_dict())
        projector_copy.zero_grad()
        z1 = projector(true_embedding)
        z2 = projector_copy(regen_embedding)

        c = z1.T @ z2

        c = c / batch_size
        c = idist.utils.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + bt_lambda * off_diag

        total_loss = (regen_lambda * loss) + (ae_lambda * ae_loss)

        outputs = dict(loss=total_loss, z1=z1, z2=z2, regen_loss=loss, ae_loss=ae_loss)

        total_loss.backward()

        for o in optimizers:
            o.step()

        return outputs

    engine = Engine(training_step)
    return engine


def byol(backbone,
         projector,
         predictor,
         ss_predictor,
         t1,
         t2,
         optimizers,
         device,
         ss_objective,
         momentum=0.996,
         ):
    target_backbone = deepcopy(backbone)
    target_projector = deepcopy(projector)
    for p in list(target_backbone.parameters()) + list(target_projector.parameters()):
        p.requires_grad = False

    def training_step(engine, batch):
        backbone.train()
        projector.train()
        predictor.train()

        for o in optimizers:
            o.zero_grad()

        x1, x2, d1, d2 = prepare_training_batch(batch, t1, t2, device)
        y1, y2 = backbone(x1), backbone(x2)
        z1, z2 = projector(y1), projector(y2)
        p1, p2 = predictor(z1), predictor(z2)
        with torch.no_grad():
            tgt1 = target_projector(target_backbone(x1))
            tgt2 = target_projector(target_backbone(x2))

        loss1 = F.cosine_similarity(p1, tgt2.detach(), dim=-1).mean().mul(-1)
        loss2 = F.cosine_similarity(p2, tgt1.detach(), dim=-1).mean().mul(-1)
        loss = (loss1 + loss2).mul(2)

        outputs = dict(loss=loss)
        outputs['z1'] = z1
        outputs['z2'] = z2

        ss_losses = ss_objective(ss_predictor, y1, y2, d1, d2)
        (loss + ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        for o in optimizers:
            o.step()

        # momentum network update
        m = 1 - (1 - momentum) * (math.cos(math.pi * (engine.state.epoch - 1) / engine.state.max_epochs) + 1) / 2
        for online, target in [(backbone, target_backbone), (projector, target_projector)]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data.mul_(m).add_(p1.data, alpha=1 - m)

        return outputs

    return Engine(training_step)


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        # idist.utils.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        # c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            # idist.utils.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def swav(backbone,
         projector: AugProjector,
         prototypes: nn.Linear,
         ss_predictor: Dict[str, nn.Module],
         t1,
         t2,
         optimizers,
         device,
         ss_objective: SSObjective,
         aug_cond,
         epsilon=0.05,
         n_iters=3,
         temperature=0.1,
         freeze_n_iters=410,
         ):
    def training_step(engine, batch):
        backbone.train()
        projector.train()
        prototypes.train()

        for o in optimizers:
            o.zero_grad()

        with torch.no_grad():
            if isinstance(prototypes, nn.parallel.DistributedDataParallel):
                p = prototypes.module
            else:
                p = prototypes

            w = p.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            p.weight.copy_(w)

        (x1, x2), (desc1, desc2), (diff1, diff2) = prepare_training_batch(batch, t1, t2, device)
        aug_ks = sorted(aug_cond)
        d1_cat = torch.cat([desc1[k] for k in aug_ks], dim=1)
        d2_cat = torch.cat([desc2[k] for k in aug_ks], dim=1)

        y1, y2 = backbone(x1), backbone(x2)
        aug_ks = sorted(aug_cond)
        d1_cat = torch.cat([desc1[k] for k in aug_ks], dim=1)
        d2_cat = torch.cat([desc2[k] for k in aug_ks], dim=1)

        z1, z2 = projector(y1, d1_cat), projector(y2, d2_cat)
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        p1, p2 = prototypes(z1), prototypes(z2)

        q1 = distributed_sinkhorn(torch.exp(p1 / epsilon).t(), n_iters)
        q2 = distributed_sinkhorn(torch.exp(p2 / epsilon).t(), n_iters)

        p1 = F.softmax(p1 / temperature, dim=1)
        p2 = F.softmax(p2 / temperature, dim=1)

        loss1 = -torch.mean(torch.sum(q1 * torch.log(p2), dim=1))
        loss2 = -torch.mean(torch.sum(q2 * torch.log(p1), dim=1))
        loss = loss1 + loss2

        outputs = dict(loss=loss)
        outputs['z1'] = z1
        outputs['z2'] = z2

        ss_losses = ss_objective(ss_predictor, y1, y2, diff1, diff2)
        (loss + ss_losses['total']).backward()
        for k, v in ss_losses.items():
            outputs[f'ss/{k}'] = v

        if engine.state.iteration < freeze_n_iters:
            for p in prototypes.parameters():
                p.grad = None

        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def collect_features(backbone,
                     dataloader,
                     device,
                     normalize=True,
                     dst=None,
                     verbose=False):
    if dst is None:
        dst = device

    backbone.eval()
    with torch.no_grad():
        features = []
        labels = []
        for i, (x, y) in enumerate(dataloader):
            if x.ndim == 5:
                _, n, c, h, w = x.shape
                x = x.view(-1, c, h, w)
                y = y.view(-1, 1).repeat(1, n).view(-1)
            z = backbone(x.to(device))
            if isinstance(z, dict):
                z = z["backbone_out"]
            if normalize:
                z = F.normalize(z, dim=-1)
            features.append(z.to(dst).detach())
            labels.append(y.to(dst).detach())
            if verbose and (i + 1) % 10 == 0:
                print(i + 1)
        features = idist.utils.all_gather(torch.cat(features, 0).detach())
        labels = idist.utils.all_gather(torch.cat(labels, 0).detach())

    return features, labels


def regen_evaluator(
        backbone, decoder,
        testloader,
        device,
        dataset: str,
        skip_connections: List[str],
        inputs_to_pool: Dict[str, int],
        decoder_input_fm_shape: Tuple[int, int, int],
    max_images: int = 5,
) -> Callable[[], plt.Figure]:
    if dataset == "stl10":
        mean = np.array([0.43, 0.42, 0.39])
        std = np.array([0.27, 0.26, 0.27])
    elif dataset == "imagenet100":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        assert False, dataset

    def evaluator():
        regen = ReGenerator(backbone, decoder, skip_connections=skip_connections, inputs_to_pool=inputs_to_pool, decoder_input_fm_shape=decoder_input_fm_shape)
        regen.eval()

        with torch.no_grad():
            for x, y in testloader:
                x = x[:max_images]
                _, _, regen_x, _ = regen(x.to(device))

                x_img = (x.cpu().numpy().transpose((0, 2, 3, 1)) * std) + mean
                r_img = (regen_x.cpu().numpy().transpose((0, 2, 3, 1)) * std) + mean

                break

            fig, ax = plt.subplots(nrows=max_images, ncols=2, figsize=(2, 2 * max_images))
            for i, (x, r) in enumerate(zip(x_img, r_img)):
                ax[i][0].imshow(x)
                ax[i][1].imshow(r)

        return fig

    return evaluator


def nn_evaluator(backbone,
                 trainloader,
                 testloader,
                 device):
    def evaluator():
        backbone.eval()
        with torch.no_grad():
            features, labels = collect_features(backbone, trainloader, device)
            corrects, total = 0, 0
            for x, y in testloader:
                z = backbone(x.to(device))
                if isinstance(z, dict):
                    z = z["backbone_out"]
                z = F.normalize(z, dim=-1)
                scores = torch.einsum('ik, jk -> ij', z, features)
                preds = labels[scores.argmax(1)]

                corrects += (preds.cpu() == y).long().sum().item()
                total += y.shape[0]
            corrects = idist.utils.all_reduce(corrects)
            total = idist.utils.all_reduce(total)

        return corrects / total

    return evaluator
