import numbers
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale

from clip_ft.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from clip_ft.utils import to_2tuple


@dataclass
class PreprocessCfg:
    image_size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0

    def __post_init__(self):
        assert self.mode in ('RGB',)

    @property
    def num_channels(self):
        return 3

    @property
    def input_size(self):
        return (self.num_channels,) + to_2tuple(self.size)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    color_jitter: Optional[Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = (0.32, 0.32, 0.32, 0.08)

    # params for simclr_jitter_gray
    color_jitter_prob: float = 0.8
    gray_scale_prob: float = 0.2


def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


def _convert_to_rgb(image):
    return image.convert('RGB')


class color_jitter(object):
    """
    Apply Color Jitter to the PIL image with a specified probability.
    """
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p=0.8):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Gray Scale to the PIL image with a specified probability.
    """
    def __init__(self, p=0.2):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = Grayscale(num_output_channels=3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def image_transform(
        train: bool,
        cfg: Optional[Union[Dict[str, Any], PreprocessCfg]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    if isinstance(cfg, dict):
        cfg = PreprocessCfg(**cfg)
    else:
        cfg = cfg or PreprocessCfg()
        
    mean = cfg.mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = cfg.std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = cfg.interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC

    resize_mode = cfg.resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)

    if train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        train_transform = [
            RandomResizedCrop(
                cfg.image_size,
                scale=aug_cfg_dict.pop('scale'),
                interpolation=interpolation_mode,
            ),
            _convert_to_rgb,
        ]
        if aug_cfg.color_jitter_prob:
            assert aug_cfg.color_jitter is not None and len(aug_cfg.color_jitter) == 4
            train_transform.extend([
                color_jitter(*aug_cfg.color_jitter, p=aug_cfg.color_jitter_prob)
            ])
        if aug_cfg.gray_scale_prob:
            train_transform.extend([
                gray_scale(aug_cfg.gray_scale_prob)
            ])
        train_transform.extend([
            ToTensor(),
            normalize,
        ])
        train_transform = Compose(train_transform)

        return train_transform
    else:
        transforms = [Resize(cfg.image_size, interpolation=interpolation_mode),
                      CenterCrop(cfg.image_size),
                      _convert_to_rgb,
                      ToTensor(),
                      normalize,
                     ]
        return Compose(transforms)


def image_transform_clip(train: bool = False, pp_cfg: Optional[PreprocessCfg] = None):
    return image_transform(train, pp_cfg)
