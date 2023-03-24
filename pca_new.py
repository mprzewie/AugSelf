import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import PCA as PCAklearn
from sklearn.utils.extmath import randomized_svd


from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path
from tqdm import tqdm

import ignite.distributed as idist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cosine_similarity

from datasets import load_datasets_for_cosine_sim
from resnets import load_backbone_out_blocks
from utils import Logger, get_engine_mock

#############
### UTILS ###
### PCA #####
#############

def pca_sklearn(X, k):
    pca = PCAklearn(n_components=k)
    pca.fit(X)
    var_exp = pca.explained_variance_ratio_
    tot = sum(var_exp)
    return var_exp, tot

def pca_randomised(X, k):
    U, S, Vh = randomized_svd(X, n_components=k, random_state=0)
    tot = sum(S**2)
    var_exp = (S**2) / tot
    return var_exp, tot

def pca_svd(X, k, full_matrices=True):
    return pca_truncated(X, k=X.shape[1], full_matrices=full_matrices)

def pca_truncated(X, k, full_matrices=True):
    X -= X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=full_matrices)
    tot = sum(S**2)
    var_exp = (S**2) / tot
    return var_exp, tot

def pca_rankk(X, k, full_matrices=True):
    X -= X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=full_matrices)
    US = (U @ S)[:k]
    Vt = Vt[:k]
    tot = sum(S**2)
    var_exp = (S**2) / tot
    return var_exp, tot

def pca_cov(X, k):
    X -= X.mean(axis=0)
    cov_mat = X.T.dot(X) / (X.shape[0]-1)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    return var_exp, tot

def pca_kernel(X, k, mode="gaussian", gamma=3):

    def phi(x1,x2):
        if mode == 'gaussian':
            return (float(np.exp(-gamma*((x1-x2).dot((x1-x2).T))))) #gaussian. (vectors are rather inconvenient in python, so instead of xTx for inner product we need to calculate xxT)
        if mode == 'polynomial':
            return (float((1 + x1.dot(x2.T))**gamma)) #polynomial
        if mode == 'hyperbolic tangent':
            return (float(np.tanh(x1.dot(x2.T) + gamma))) #hyperbolic tangent

    Kernel=[]
    for x in X.T:
        xi=np.mat(x)
        row=[]
        for y in X.T:
            xj=np.mat(y)
            kf=phi(xi,xj)
            row.append(kf)
        Kernel.append(row)
    kernel=np.array(Kernel)

    # Centering the symmetric NxN kernel matrix.
    N = kernel.shape[0]
    one_n = np.ones((N,N)) / N
    kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n) #centering

    eig_vals, eig_vecs = np.linalg.eigh(kernel) #the eigvecs are sorted in ascending eigenvalue order.
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    return var_exp, tot
  

#############
#############

def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()

    logdir = Path(args.ckpt).parent

    args.origin_run_name = logdir.name

    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"pca-{args.dataset}", args=args,
        job_type="eval_feature_pca"

    )

    # DATASETS

    datasets = load_datasets_for_cosine_sim(
        dataset=args.dataset,
        pretrain_data=args.pretrain_data,
        datadir=args.datadir
    )

    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)

    testloader  = build_dataloader(datasets['test'],  drop_last=False)
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

    metrics = dict()

    with torch.no_grad():
        for i, (X, _) in enumerate(tqdm(testloader)):
            X_transformed = {
                t_name: t(X) for (t_name, t) in transforms_dict.items()
            }
            X_norm = X_transformed["identity"]

            bs = X_norm.shape[0]

            feats_norm = backbone(X_norm.to(device))

            for t_name, X_t in X_transformed.items():
                feats_t = backbone(X_t.to(device))
                assert feats_norm.keys() == feats_t.keys()

                for block_name, fn in feats_norm.items():
                    ft = feats_t[block_name]
                    ft_r = ft.reshape(bs, -1).detach().cpu().numpy()

                    # selection
                    if block_name not in ["out"]: # "l4", 
                        continue

                    # PCA

                    possible = ["sklearn", "randomised", "svd", "truncated", "rankk", "kernel", "cov"]

                    if args.pca_type == "sklearn":
                        var_exp, tot = pca_sklearn(ft_r, args.k)
                    elif args.pca_type == "randomised":
                        var_exp, tot = pca_randomised(ft_r, args.k)
                    elif args.pca_type == "svd":
                        var_exp, tot = pca_svd(ft_r, args.k, args.full_matrices)
                    elif args.pca_type == "truncated":
                        var_exp, tot = pca_truncated(ft_r, args.k, args.full_matrices)
                    elif args.pca_type == "rankk":
                        var_exp, tot = pca_rankk(ft_r, args.k, args.full_matrices)
                    elif args.pca_type == "kernel":
                        var_exp, tot = pca_kernel(ft_r, args.k, args.pca_kernel)
                    elif args.pca_type == "cov":
                        var_exp, tot = pca_cov(ft_r, args.k)

                    cum_var_exp = np.cumsum(var_exp)
                    
                    #Plot both the individual variance explained and the cumulative:
                    plt.clf()
                    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center', label='individual explained variance')
                    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
                    plt.ylabel('Explained variance ratio')
                    plt.xlabel('Principal components')
                    plt.legend(loc='best')
                    plt.title(f"PCA {args.pca_type} {t_name} {block_name} expl {tot}")
                    # logger.log(engine=engine_mock, global_step=i, pca={f"feature_pca_{args.dataset}_{t_name}_{block_name}_{i}": plt}) #TODO
                    # logger.log({f"feature_pca_{args.dataset}_{t_name}_{block_name}_{i}": plt})
                    plt.savefig(f"pca/feature_pca_{args.pca_type}_{args.dataset}_{t_name}_{block_name}_{i}.png")

                    logger.log_msg(f'variance explainability is {cum_var_exp[-1]}')
                    metrics[f"feature_pca/{args.dataset}/{t_name}/{block_name}/{i}"] = cum_var_exp[-1]
                    logger.log(
                        engine=engine_mock, global_step=i,
                        **metrics
                    )

                    #if (i+1) % args.print_freq == 0:
                    #    logger.log_msg(
                    #        f'{i + 1:3d} | {block_name} | {t_name} | pos: {np.mean(t_name_to_b_name_to_positive_sims[t_name][block_name]):.4f} | neg: {np.mean(t_name_to_b_name_to_negative_sims[t_name][block_name]):.4f})'
                    #    )

        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrain-data', type=str, default='imagenet100')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--pca-type', type=str, default='sklearn')
    parser.add_argument('--full-matrices', action='store_true')
    parser.add_argument('--no-full-matrices', dest='full-matrices', action='store_false')
    parser.add_argument('--pca-kernel', type=str, default='gaussian')
    parser.add_argument('--k', type=int, default=512)
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)