from copy import deepcopy

from torch import nn
from torch.nn import functional as tnnf
from resnets import ResnetOutBlocks
from decoders import LightDecoder
class ReGenerator(nn.Module):

    def __init__(
        self,
        backbone: ResnetOutBlocks,
        decoder: LightDecoder,

    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.backbone_copy = deepcopy(self.backbone)

    def reset_backbone_copy(self):
        self.backbone_copy.load_state_dict(self.backbone.state_dict())
        self.backbone_copy.zero_grad()

    def forward(self, X, reset_backbone_copy: bool = True):

        if reset_backbone_copy:
            self.reset_backbone_copy()

        e = self.backbone(X)
        true_embedding = e["backbone_out"]
        regen_X = self.decoder([e["l4"], e["l3"], e["l2"], e["l1"]])
        assert X.shape == regen_X.shape, (X.shape, regen_X.shape)
        regen_embedding = self.backbone_copy(regen_X)["backbone_out"]
        assert true_embedding.shape == regen_embedding.shape, (true_embedding.shape, regen_embedding.shape)
        mse_loss = tnnf.mse_loss(X, regen_X)

        return true_embedding, regen_embedding, regen_X, mse_loss



