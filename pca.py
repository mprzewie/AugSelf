import torch
import numpy as np
import matplotlib.pyplot as plt

class All_PCA:
    def __init__(self, Data, type='deterministic'):
        '''
        deterministic | truncated | probabilistic
        '''
        self.Data = Data
        self.type = type
    
    def __repr__(self):
        return f'PCA({self.Data})'
    
    @staticmethod
    def center(Data):
        #Convert to torch Tensor and keep the number of rows and columns
        # t = torch.from_numpy(self.Data)
        t = Data
        no_rows, no_columns = t.size()
        row_means = torch.mean(t, 1)
        #Expand the matrix in order to have the same shape as X and substract, to center
        X = t - row_means.view(no_rows, 1)
        #for_subtraction = row_means.repeat(no_rows, no_columns)
        #X = t - for_subtraction #centered
        return X

    @classmethod
    def decomposition(cls, Data, k, type='deterministic'):
        X = cls.center(Data)
        if type == 'deterministic':
            U,S,V = torch.svd(X)
        elif type == 'truncated':
            U,S,V = torch.svd_lowrank(X, q = np.sqrt(len(X[0])))
        eigvecs = U.t()[:,:k] #the first k vectors will be kept
        y = torch.mm(U,eigvecs)

        #Save variables to the class object, the eigenpair and the centered data
        cls.eigenpair = (eigvecs, S)
        cls.centred_data=X
        return y

    def explained_variance(self):
        #Total sum of eigenvalues (total variance explained)
        tot = sum(self.eigenpair[1])
        #Variance explained by each principal component
        var_exp = [(i / tot) for i in sorted(self.eigenpair[1], reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        #X is the centered data
        X = All_PCA.Data
        #Plot both the individual variance explained and the cumulative:
        plt.bar(range(X.size()[1]), var_exp, alpha=0.5, align='center', label='individual explained variance')
        plt.step(range(X.size()[1]), cum_var_exp, where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.show()

def Kernel_PCA(X, gamma=3, dims=1, mode='gaussian'):
    '''
    X is the necessary input. The data.
    gamma will be the user defined value that will be used in the kernel functions. The default is 3.
    dims will be the number of dimensions of the final output (basically the number of components to be picked). The default is 1.
    mode has three options 'gaussian', 'polynomial', 'hyperbolic tangent' which will be the kernel function to be used. The default is gaussian.
    '''

    print('Now running Kernel PCA with', mode, 'kernel function...')

    #First the kernel function picked by the user is defined. Vectors need to be input in np.mat type

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

    eigVals, eigVecs = np.linalg.eigh(kernel) #the eigvecs are sorted in ascending eigenvalue order.
    y=eigVecs[:,-dims:].T #user defined dims, since the order is reversed, we pick principal components from the last columns instead of the first
    return (y)

#TODO
class Probabilistic_PCA():
    def __init__(self) -> None:
        pass

from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import PCA as PCAklearn


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
                    ft_r = ft.reshape(bs, -1)

                    # selection
                    if block_name != "out":
                        continue

                    # PCA

                    if args.buildin:
                        print("--sklearn PCA")
                        pca_t = PCAklearn(n_components=args.k)
                        pca_t.fit(X)
                        var_exp = pca_t.explained_variance_ratio_
                        tot = sum(var_exp)
                    else:
                        print(f"--SVD based PCA {args.pca_type}")
                        pca = All_PCA(ft_r, type=args.pca_type)
                        pca.decomposition(ft_r, args.k, type=args.pca_type)
                        tot = sum(pca_t.eigenpair[1])
                        #Variance explained by each principal component
                        var_exp = [(i / tot) for i in sorted(pca_t.eigenpair[1], reverse=True)]

                    cum_var_exp = np.cumsum(var_exp)
                    
                    #Plot both the individual variance explained and the cumulative:
                    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center', label='individual explained variance')
                    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
                    plt.ylabel('Explained variance ratio')
                    plt.xlabel('Principal components')
                    plt.legend(loc='best')
                    # logger.log(engine=engine_mock, global_step=i, pca={f"feature_pca_{args.dataset}_{t_name}_{block_name}_{i}": plt}) #TODO
                    # logger.log({f"feature_pca_{args.dataset}_{t_name}_{block_name}_{i}": plt})
                    plt.savefig(f"feature_pca_{args.dataset}_{t_name}_{block_name}_{i}.png")

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
    parser.add_argument('--buildin', action='store_true')
    parser.add_argument('--pca_type', type=str, default='deterministic')
    parser.add_argument('--k', type=int, default=32)
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)