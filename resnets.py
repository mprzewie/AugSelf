from functools import partial
from typing import Any, Type, Union, List, Dict

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls

from models import reset_parameters
from vits import VisionTransformerMoCo, _cfg


class ResnetOutBlocks(ResNet):
    def _forward_impl(self, x: Tensor) -> Dict[str, Tensor]:
        # See note [TorchScript super()]
        in_x=x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_out = x = self.maxpool(x)

        l1 = x = self.layer1(x)
        l2 = x = self.layer2(x)
        l3 = x = self.layer3(x)
        l4 = x = self.layer4(x)

        x = self.avgpool(x)
        backbone_out = x = torch.flatten(x, 1)
        x = self.fc(x)

        return dict(
            input=in_x,
            conv1=conv1_out,
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            backbone_out = backbone_out,
            out=x
        )

class VitOutBlocks(VisionTransformerMoCo):
    def forward(self, x):

        result_dict = dict()
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, b in enumerate(self.blocks):
            x = b(x)
            result_dict[f"b{i}"] = x[:, 0]
        # x = self.blocks(x)
        x = self.norm(x)
        result_dict["norm"] = x[:, 0]

        assert self.dist_token is None
        x = self.pre_logits(x[:, 0])
        result_dict["backbone_out"] = x
        # else:
        #     assert False
        #     x = x[:, 0], x[:, 1]

        # x = self.forward_features(x)
        assert self.head_dist is None
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        #     if self.training and not torch.jit.is_scripting():
        #         # # during inference, return the average of both classifier predictions
        #         # return x, x_dist
        #         raise NotImplementedError("only inference allowed!")
        #     else:
        #         x = (x + x_dist) / 2
        #
        # else:
        x = self.head(x)

        result_dict["out"] = x
        return result_dict



def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResnetOutBlocks:
    model = ResnetOutBlocks(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResnetOutBlocks:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResnetOutBlocks:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def vit_small(stop_grad_conv1: bool=True, num_classes: int=4096, **kwargs):
    model = VitOutBlocks(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), stop_grad_conv1=stop_grad_conv1, num_classes=num_classes, **kwargs)
    model.default_cfg = _cfg()
    return model

def load_backbone_out_blocks(args):
    name = args.model

    if name == "resnet18":
        backbone = resnet18(zero_init_residual=True)
    elif name == "resnet50":
        backbone = resnet50(zero_init_residual=True)

    elif name == "vit_small":
        backbone =vit_small()
        args.num_backbone_features = backbone.head.weight.shape[1]
        backbone.head = nn.Identity()
        return backbone
    else:
        raise NotImplementedError(name)

    if name.startswith('cifar_'):
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    args.num_backbone_features = backbone.fc.weight.shape[1]
    backbone.fc = nn.Identity()
    reset_parameters(backbone)
    return backbone