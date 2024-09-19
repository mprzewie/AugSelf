from copy import deepcopy
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as tnnf
from resnets import ResnetOutBlocks
from decoders import LightDecoder

class Pooler(nn.Module):
    def __init__(
        self,
        inputs_to_pool: Dict[str, int],
        decoder_input_fm_shape: Tuple[int, int, int]
    ):
        super().__init__()
        c, fmh, fmw = decoder_input_fm_shape
        self.inputs_to_pool = inputs_to_pool
        self.decoder_input_fm_shape = decoder_input_fm_shape

        self.net = nn.Linear(sum(inputs_to_pool.values()), c * fmh * fmw)

    def forward(self, embeddings: Dict[str, torch.Tensor]):
        _X = embeddings[list(embeddings.keys())[0]]
        batch_size = len(_X)
        pooled_outputs = [torch.zeros(batch_size, 0).to(_X.device)]

        for layer_id in self.inputs_to_pool:
            o = embeddings[layer_id].mean(dim=(2, 3))
            pooled_outputs.append(o)

        pooler_inputs = torch.cat(pooled_outputs, dim=1)
        pooler_outputs = self.net(pooler_inputs)
        if pooler_inputs.shape[1] == 0:
            pooler_outputs = pooler_outputs.unsqueeze(-1).repeat(batch_size, 1)

        pooler_outputs = pooler_outputs.reshape(
            batch_size, *self.decoder_input_fm_shape
        )
        return pooler_outputs


class ReGenerator(nn.Module):

    def __init__(
        self,
        backbone: ResnetOutBlocks,
        decoder: LightDecoder,
        pooler: Pooler,
        skip_connections: List[str],
        backbone_input: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.pooler = pooler
        self.backbone_copy = deepcopy(self.backbone)
        self.skip_connections = skip_connections
        self.backbone_input = backbone_input

    def reset_backbone_copy(self):
        self.backbone_copy.load_state_dict(self.backbone.state_dict())
        self.backbone_copy.zero_grad()

    def forward(self, X, reset_backbone_copy: bool = True):

        batch_size = len(X)
        if reset_backbone_copy:
            self.reset_backbone_copy()

        if self.backbone_input == "real":
            true_embedding = self.backbone(X)
        else:
            true_embedding = self.backbone_copy(X)
            true_embedding = {
                k: v.detach()
                for (k,v) in true_embedding.items()
            }

        pooler_outputs = self.pooler(true_embedding)

        if "l4" in self.skip_connections:
            pooler_outputs = pooler_outputs + true_embedding["l4"]

        decoder_inputs = [pooler_outputs]

        for s in ["l3", "l2", "l1"]:
            if s in self.skip_connections:
                decoder_inputs.append(true_embedding[s])
            else:
                decoder_inputs.append(None)

        regen_X = self.decoder(decoder_inputs)

        assert X.shape == regen_X.shape, (X.shape, regen_X.shape)

        if self.backbone_input == "real":
            regen_embedding = self.backbone_copy(regen_X)
        else:
            regen_embedding = self.backbone(X)

        assert all([true_embedding[k].shape==regen_embedding[k].shape for k in true_embedding.keys()]), (
            {k: v.shape for (k,v) in true_embedding.items()},
            {k: v.shape for (k,v) in regen_embedding.items()},
        )
        mse_loss = tnnf.mse_loss(X, regen_X)

        return true_embedding, regen_embedding, regen_X, mse_loss




