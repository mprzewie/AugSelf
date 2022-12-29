import torch
from torch import nn as nn

from models import load_mlp

AUG_DESC_SIZE_CONFIG = {
    "crop": 4,
    "color": 4,
    "flip": 1,
    "blur": 1,
    # "rot": 4,
    # "sol": 1,
    "grayscale": 1
}


class AUG_TREATMENT:
    raw = "raw"
    mlp = "mlp"
    hn = "hn"


class AugProjector(nn.Module):
    def __init__(
            self,  args, proj_out_dim: int, proj_depth: int = 2
    ):
        super().__init__()
        self.aug_treatment = args.aug_treatment
        self.aug_nn_depth = args.aug_nn_depth
        self.aug_nn_width = args.aug_nn_width


        if self.aug_treatment == AUG_TREATMENT.raw:
            self.num_aug_features = sum(AUG_DESC_SIZE_CONFIG.values())

            self.aug_processor = nn.Identity()

        elif self.aug_treatment == AUG_TREATMENT.mlp:
            self.num_aug_features = self.aug_nn_width

            self.aug_processor = load_mlp(
                n_in=sum(AUG_DESC_SIZE_CONFIG.values()),
                n_hidden=self.aug_nn_width,
                n_out=self.aug_nn_width,
                num_layers=self.aug_nn_depth
            )
            print(self.aug_processor)

        elif self.aug_treatment == AUG_TREATMENT.hn:
            raise NotImplementedError("Hypernetworks: TODO")
            self.num_aug_features = 0


        if self.aug_treatment in [AUG_TREATMENT.raw, AUG_TREATMENT.mlp]:

            self.projector = load_mlp(
                args.num_backbone_features + self.num_aug_features,
                args.num_backbone_features,
                proj_out_dim,
                num_layers=proj_depth,
                last_bn=False
            )

    def forward(self, x: torch.Tensor, aug_desc: torch.Tensor):

        if self.aug_treatment in [AUG_TREATMENT.mlp, AUG_TREATMENT.raw]:
            aug_desc = self.aug_processor(aug_desc)
            concat = torch.concat([x, aug_desc], dim=1)
            return self.projector(concat)


        elif self.aug_treatment == AUG_TREATMENT.hn:
            ...
