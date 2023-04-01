conda create -n AugSelfConditioning python=3.8
conda activate AugSelfConditioning
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
# sanity check torch.cuda.is_available()
pip install scipy tensorboard kornia==0.4.1 sklearn
conda install -c conda-forge packaging
conda install -c conda-forge wandb
conda install ignite -c pytorch