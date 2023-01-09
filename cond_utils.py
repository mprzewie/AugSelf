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

class AUG_HN_TYPES:
    mlp = "mlp"
    mlp_bn = "mlp_bn"


class AugProjector(nn.Module):
    def __init__(
            self, args, proj_out_dim: int, proj_depth: int = 2
    ):
        super().__init__()
        self.aug_treatment = args.aug_treatment
        self.aug_hn_type = args.aug_hn_type
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
            num_weights_to_generate = 0

            layer_in_size = args.num_backbone_features
            layer_out_size = args.num_backbone_features

            self.layers_config = []
            for i in range(proj_depth):
                if i == (proj_depth - 1):
                    layer_out_size = proj_out_dim

                print(f"HN for proj layer #{i}")
                print(f"{layer_in_size=}")
                print(f"{layer_out_size=}")
                print(f"weight_params: {layer_in_size * layer_out_size}")
                print(f"bias_params: {layer_out_size}")
                print("-------")

                layer_cfg = dict(
                    n_in=layer_in_size,
                    n_out=layer_out_size,
                    weight_range=(
                        num_weights_to_generate,
                        num_weights_to_generate + layer_in_size * layer_out_size
                    ),
                    bias_range=(
                        num_weights_to_generate + layer_in_size * layer_out_size,
                        num_weights_to_generate + (layer_in_size * layer_out_size) + layer_out_size
                    )
                )
                self.layers_config.append(layer_cfg)

                num_weights_to_generate += layer_in_size * layer_out_size  # weights
                num_weights_to_generate += layer_out_size  # biases

                layer_in_size = layer_out_size

            self.projector_hn = load_mlp(
                n_in=sum(AUG_DESC_SIZE_CONFIG.values()),
                n_hidden=self.aug_nn_width,
                n_out=num_weights_to_generate,
                num_layers=self.aug_nn_depth
            )
            print("HN layer config")
            from pprint import pprint
            pprint(self.layers_config)

            if self.aug_hn_type == AUG_HN_TYPES.mlp_bn:
                self.projector_bns = nn.ModuleList([
                    nn.BatchNorm1d(l_cfg["n_out"])
                    for l_cfg in self.layers_config[:-1]
                ])
                print("Projector Batchnorms")
                print(self.projector_bns)


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
            generated_weights = self.projector_hn(aug_desc)

            b_s, x_s = x.shape

            x_proc = x.reshape(b_s, 1, x_s)
            for l, l_cfg in enumerate(self.layers_config):
                w_r_s, w_r_e = l_cfg["weight_range"]
                b_r_s, b_r_e = l_cfg["bias_range"]

                w_in = l_cfg["n_in"]
                w_out = l_cfg["n_out"]

                w_weights = generated_weights[:, w_r_s:w_r_e].reshape(b_s, w_in, w_out)
                b_weights = generated_weights[:, b_r_s:b_r_e].reshape(b_s, 1, w_out)

                x_proc = torch.bmm(x_proc, w_weights)
                x_proc = x_proc + b_weights
                # TODO - batchnorm?

                if l != (len(self.layers_config) - 1):

                    if self.aug_hn_type == AUG_HN_TYPES.mlp_bn:
                        b_s, _, x_p_s = x_proc.shape
                        x_proc = self.projector_bns[l](x_proc.reshape(b_s, x_p_s)).reshape(b_s, 1, x_p_s)

                    x_proc = torch.relu(x_proc)

            # print(f"{x_proc.shape=}")
            return x_proc.reshape(b_s, -1)
