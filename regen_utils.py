from copy import deepcopy

from torch import nn


class ReGenerator(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,

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

        true_embedding = self.backbone(X)
        regen_X = self.decoder(true_embedding)
        assert X.shape == regen_X.shape, (X.shape, regen_X.shape)
        regen_embedding = self.backbone_copy(regen_X)
        assert true_embedding.shape == regen_embedding.shape, (true_embedding.shape, regen_embedding.shape)
        ae_loss = ((X - regen_X) ** 2).mean()

        return true_embedding, regen_embedding, regen_X, ae_loss



