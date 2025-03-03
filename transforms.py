import random
from typing import Union, Tuple, List, Optional, Dict
from typing import Union, Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as NF
import torchvision.transforms as T
import torchvision.transforms.functional as F
import kornia
import kornia.augmentation as K
from kornia import adjust_saturation, adjust_hue, pi
from kornia.augmentation.utils import _transform_input, _validate_input_dtype


import kornia.augmentation.functional as KF
from torchvision.transforms import Normalize


class MultiView:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class RandomResizedCrop(T.RandomResizedCrop):
    def forward(self, img):
        W, H = F.get_image_size(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        tensor = F.to_tensor(img) if not isinstance(img, torch.Tensor) else img
        return tensor, torch.tensor([i/H, j/W, h/H, w/W], dtype=torch.float)


def apply_adjust_brightness(img1, params):
    ratio = params['brightness_factor'][:, None, None, None].to(img1.device)
    img2 = torch.zeros_like(img1)
    return (ratio * img1 + (1.0-ratio) * img2).clamp(0, 1)


def apply_adjust_contrast(img1, params):
    ratio = params['contrast_factor'][:, None, None, None].to(img1.device)
    img2 = 0.2989 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
    img2 = torch.mean(img2, dim=(-2, -1), keepdim=True)
    return (ratio * img1 + (1.0-ratio) * img2).clamp(0, 1)


class ColorJitter(K.ColorJitter):
    def apply_transform(self, x, params, transform: Optional[torch.Tensor] = None):
        transforms = [
            lambda img: apply_adjust_brightness(img, params),
            lambda img: apply_adjust_contrast(img, params),
            lambda img: KF.apply_adjust_saturation(img, params),
            lambda img: KF.apply_adjust_hue(img, params)
        ]

        x_new = x
        for idx in params['order'].tolist():
            t = transforms[idx]
            x_new = t(x_new)

        x_means = x.mean(dim=[2,3])
        x_new_means = x_new.mean(dim=[2,3])
        params["x_means_diff"] = (x_means - x_new_means).cpu()

        return x_new


class GaussianBlur(K.AugmentationBase2D):
    def __init__(self, kernel_size, sigma, border_type='reflect',
                 return_transform=False, same_on_batch=False, p=0.5):
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def __repr__(self):
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape):
        return dict(sigma=torch.zeros(batch_shape[0]).uniform_(self.sigma[0], self.sigma[1]))

    def apply_transform(self, input, params, transform: Optional[torch.Tensor] = None):
        sigma = params['sigma'].to(input.device)
        k_half = self.kernel_size // 2
        x = torch.linspace(-k_half, k_half, steps=self.kernel_size, dtype=input.dtype, device=input.device)
        pdf = torch.exp(-0.5*(x[None, :] / sigma[:, None]).pow(2))
        kernel1d = pdf / pdf.sum(1, keepdim=True)
        kernel2d = torch.bmm(kernel1d[:, :, None], kernel1d[:, None, :])
        input = NF.pad(input, (k_half, k_half, k_half, k_half), mode=self.border_type)
        input = NF.conv2d(input.transpose(0, 1), kernel2d[:, None], groups=input.shape[0]).transpose(0, 1)
        return input


class RandomRotation(K.AugmentationBase2D):
    def __init__(self, return_transform=False, same_on_batch=False, p=0.5):
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)

    def __repr__(self):
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape):
        degrees = torch.randint(0, 4, (batch_shape[0], ))
        return dict(degrees=degrees)

    def apply_transform(self, input, params, transform: Optional[torch.Tensor] = None):
        degrees = params['degrees']
        input = torch.stack([torch.rot90(x, k, (1, 2)) for x, k in zip(input, degrees.tolist())], 0)
        return input

class KRandomResizedCrop(K.RandomResizedCrop):
    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return KF.apply_crop(input, params, self.flags)


def _extract_w(t):
    if isinstance(t, GaussianBlur):
        m = t._params['batch_prob']
        w = torch.zeros(m.shape[0], 1)
        w[m] = t._params['sigma'].unsqueeze(-1)
        return w

    elif isinstance(t, ColorJitter):
        to_apply = t._params['batch_prob']
        w = torch.zeros(to_apply.shape[0], 4)
        w[to_apply, 0] = (t._params['brightness_factor'] - 1) / (t.brightness[1]-t.brightness[0])
        w[to_apply, 1] = (t._params['contrast_factor'] - 1) / (t.contrast[1]-t.contrast[0])
        w[to_apply, 2] = (t._params['saturation_factor'] - 1) / (t.saturation[1]-t.saturation[0])
        w[to_apply, 3] = t._params['hue_factor'] / (t.hue[1]-t.hue[0])

        x_means_batch = torch.zeros(to_apply.shape[0], 3)
        if "x_means_diff" in t._params:
            x_means_diff = t._params["x_means_diff"]
            x_means_batch[to_apply] = x_means_diff

        return w, x_means_batch

    elif isinstance(t, RandomRotation):
        to_apply = t._params['batch_prob']
        w = torch.zeros(to_apply.shape[0], dtype=torch.long)
        w[to_apply] = t._params['degrees']
        return w

    elif isinstance(t, K.RandomSolarize):
        to_apply = t._params['batch_prob']
        w = torch.ones(to_apply.shape[0])
        w[to_apply] = t._params['thresholds_factor']
        return w

    elif isinstance(t, K.RandomGrayscale):
        w = t._params["batch_prob"].float().reshape(-1, 1)
        return w

def extract_diff(transforms1, transforms2, crop1, crop2):
    diff = {}
    for t1, t2 in zip(transforms1, transforms2):
        if isinstance(t1, K.RandomHorizontalFlip):
            f1 = t1._params['batch_prob']
            f2 = t2._params['batch_prob']
            break

    center1 = crop1[:, :2]+crop1[:, 2:]/2
    center2 = crop2[:, :2]+crop2[:, 2:]/2
    center1[f1, 1] = 1-center1[f1, 1]
    center2[f1, 1] = 1-center2[f1, 1]
    diff['crop'] = torch.cat([center1-center2, crop1[:, 2:]-crop2[:, 2:]], 1)
    diff['flip'] = (f1==f2).float().unsqueeze(-1)
    for t1, t2 in zip(transforms1, transforms2):
        if isinstance(t1, K.RandomHorizontalFlip):
            pass

        elif isinstance(t1, K.RandomGrayscale):
            pass

        elif isinstance(t1, GaussianBlur):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['blur'] = w1-w2

        elif isinstance(t1, K.Normalize):
            pass

        elif isinstance(t1, ColorJitter):
            w1, x_means_diff_1 = _extract_w(t1)
            w2, x_means_diff_2 = _extract_w(t2)
            diff['color'] = w1-w2
            diff["color_diff"] = x_means_diff_1 - x_means_diff_2

        elif isinstance(t1, (nn.Identity, nn.Sequential)):
            pass

        elif isinstance(t1, RandomRotation):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['rot'] = (w1-w2+4) % 4

        elif isinstance(t1, K.RandomSolarize):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            diff['sol'] = w1-w2

        else:
            raise Exception(f'Unknown transform: {str(t1.__class__)}')

    return diff


def extract_aug_descriptors(
    transforms1, crop1
):
    results = {}
    f1 = None
    for t1 in transforms1:
        if isinstance(t1, K.RandomHorizontalFlip):
            f1 = t1._params['batch_prob']
            break

    if f1 is None:
        f1 = torch.zeros(len(crop1)).bool()

    center1 = crop1[:, :2] + crop1[:, 2:] / 2
    center1[f1, 1] = 1 - center1[f1, 1]
    results['crop'] = torch.cat([center1, crop1[:, 2:]], 1)
    results['flip'] = f1.float().unsqueeze(-1)
    for t1 in transforms1:
        if isinstance(t1, K.RandomHorizontalFlip):
            pass

        elif isinstance(t1, K.RandomGrayscale):
            results["grayscale"] = _extract_w(t1)

        elif isinstance(t1, GaussianBlur):
            w1 = _extract_w(t1)
            results['blur'] = w1

        elif isinstance(t1, K.Normalize):
            pass

        elif isinstance(t1, ColorJitter):
            w1, x_means_batch = _extract_w(t1)
            results['color'] = w1
            results["color_diff"] = x_means_batch

        elif isinstance(t1, (nn.Identity, nn.Sequential)):
            pass

        elif isinstance(t1, RandomRotation):
            w1 = _extract_w(t1)
            results['rot'] = (w1 + 4) % 4

        elif isinstance(t1, K.RandomSolarize):
            w1 = _extract_w(t1)
            results['sol'] = w1
        elif isinstance(t1, Normalize):
            pass
        else:
            raise Exception(f'Unknown transform: {str(t1.__class__)}')

    return results