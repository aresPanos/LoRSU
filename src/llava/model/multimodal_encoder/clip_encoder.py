import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.parametrize as parametrize

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from open_clip import load_openai_model, CLIP


def create_llava_vision_tower_dict(clim_model: CLIP) -> Dict[str, torch.Tensor]:
    openclip_ckpt = clim_model.state_dict()
    dict_openclip2llava = {'visual.class_embedding': 'vision_model.embeddings.class_embedding',
                       'visual.positional_embedding': 'vision_model.embeddings.position_embedding.weight',
                       'visual.conv1.weight': 'vision_model.embeddings.patch_embedding.weight',
                       'visual.ln_pre.weight': 'vision_model.pre_layrnorm.weight',
                       'visual.ln_pre.bias': 'vision_model.pre_layrnorm.bias',
                       'visual.ln_post.weight': 'vision_model.post_layernorm.weight',
                       'visual.ln_post.bias': 'vision_model.post_layernorm.bias',
                       }
    state_dict_updated = OrderedDict()
    for param_name_opeclip in dict_openclip2llava.keys():
        state_dict_updated[dict_openclip2llava[param_name_opeclip]] = openclip_ckpt[param_name_opeclip]
        
    num_layers = 24
    dict_mlp_llava2openclip = {'fc1': 'c_fc', 'fc2': 'c_proj', 'layer_norm': 'ln_'}
    for layer in range(num_layers):
        for weight_bias in ['weight', 'bias']:
            q_openclip, k_openclip, v_openclip = openclip_ckpt[f'visual.transformer.resblocks.{layer}.attn.in_proj_{weight_bias}'].chunk(3)
            state_dict_updated[f'vision_model.encoder.layers.{layer}.self_attn.q_proj.{weight_bias}'] = q_openclip
            state_dict_updated[f'vision_model.encoder.layers.{layer}.self_attn.k_proj.{weight_bias}'] = k_openclip
            state_dict_updated[f'vision_model.encoder.layers.{layer}.self_attn.v_proj.{weight_bias}'] = v_openclip
            
            state_dict_updated[f'vision_model.encoder.layers.{layer}.self_attn.out_proj.{weight_bias}'] = openclip_ckpt[f'visual.transformer.resblocks.{layer}.attn.out_proj.{weight_bias}']            
            
            for module_number in range(2):
                state_dict_updated[f'vision_model.encoder.layers.{layer}.layer_norm{module_number+1}.{weight_bias}'] = openclip_ckpt[f'visual.transformer.resblocks.{layer}.ln_{module_number+1}.{weight_bias}']
                state_dict_updated[f'vision_model.encoder.layers.{layer}.mlp.fc{module_number+1}.{weight_bias}'] = openclip_ckpt['visual.transformer.resblocks.{}.mlp.{}.{}'.format(layer, dict_mlp_llava2openclip[f'fc{module_number+1}'], weight_bias)]    
           
    return state_dict_updated
        
        
def transpose(weight: torch.Tensor, fan_in_fan_out: bool) -> torch.Tensor:
    return weight.T if fan_in_fan_out else weight

class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class LoRAParametrization(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int = 4, 
                 lora_alpha: float = 1.0):
        super().__init__()
        assert rank > 0, f"Rank should be a positive integer but rank={rank} was given."
        self.lora_A = nn.Parameter(torch.zeros((out_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, in_features)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.forward_fn = self.lora_forward

    def lora_forward(self, X: torch.Tensor):
        return X + (self.lora_A@self.lora_B) * self.scaling

    def forward(self, X: torch.Tensor):
        return self.forward_fn(X)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank: int = 4, lora_alpha: float = 1.0):
        assert isinstance(layer, nn.Linear), f"`layer` must be a nn.Linear instance but {type(layer)} was received."
        return cls(layer.in_features, layer.out_features, rank, lora_alpha).to(device=layer.weight.device)
    
    @classmethod
    def from_mhattn(cls, layer: nn.MultiheadAttention, rank: int = 4, lora_alpha: float = 1.0):
        assert isinstance(layer, nn.MultiheadAttention), f"`layer` must be a nn.MultiheadAttention instance but {type(layer)} was received."
        out_features, in_features = layer.in_proj_weight.shape
        return cls(in_features, out_features, rank, lora_alpha).to(device=layer.in_proj_weight.device)    


def apply_lora_openai(layer: nn.Module, 
                      rank: int = 2, 
                      lora_alpha: float = 1.0, 
                      register: bool = True, 
                      merge: bool = False):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        if isinstance(layer, nn.Linear):
            parametrize.register_parametrization(layer, "weight", LoRAParametrization.from_linear(layer, rank, lora_alpha))
        elif isinstance(layer, nn.MultiheadAttention):
            parametrize.register_parametrization(layer, "in_proj_weight", LoRAParametrization.from_mhattn(layer, rank, lora_alpha))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_lora_openai(model: nn.Module, rank: int, lora_alpha: float = 1.0):
    """add lora parametrization to all layers in a model. Calling it twice will add lora twice"""
    model.apply(partial(apply_lora_openai, rank=rank, lora_alpha=lora_alpha))
            
                
def add_lora_visual_openai(model: CLIP, rank: int, lora_alpha: float = 1.0):
    assert isinstance(model, CLIP), f"The model should an instance of class CLIP but {type(model)} was given."
    """add LoRA parametrization to all layers in the visual encoder of CLIP. Calling it twice will add lora twice"""
    add_lora_openai(model.visual, rank, lora_alpha)
    
    out_features, in_features = model.visual.proj.shape
    parametrize.register_parametrization(model.visual, "proj", LoRAParametrization(in_features, out_features, rank, lora_alpha).to(device=model.visual.proj.device))
    
    
def merge_lora(model: CLIP):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora_openai, register=False, merge=True))
    
    
def get_submodules(model: nn.Module, key: str):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name   
    
    
def extract_lora_rank_alpha_from_path(model_path: str) -> Tuple[int, int]:
    rank_alpha = model_path.split('/')[-3].split('_')
    rank,  alpha = [int(item.split('-')[1]) for item in rank_alpha]
    
    return rank, alpha


def extract_adalora_rank_alpha_from_path(model_path: str) -> Tuple[int, int]:
    rank_alpha = model_path.split('/')[-3].split('_')
    rank,  alpha = [int(item.split('-')[1]) for item in [rank_alpha[0], rank_alpha[2]]]
    
    return rank, alpha


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            
    def load_model(self, fine_tuned_clip_model: Optional[str] = None, device_map: Optional[str] = None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)        
        if fine_tuned_clip_model:
            print(f'Use fine-tuned {self.vision_tower_name} from file `{fine_tuned_clip_model}`.')
            pretrained_clip = load_openai_model(name='ViT-L-14-336', precision='fp32', device=self.device)
            loaded_dict = torch.load(fine_tuned_clip_model, map_location=self.device)['state_dict']
            if 'lora' in fine_tuned_clip_model:
                lora_rank, lora_alpha = extract_lora_rank_alpha_from_path(fine_tuned_clip_model)
                add_lora_visual_openai(pretrained_clip, lora_rank, lora_alpha)
                pretrained_clip.load_state_dict(loaded_dict, strict=False)
                merge_lora(pretrained_clip)
            elif 'LoRSU' in fine_tuned_clip_model:
                pretrained_clip.visual.load_state_dict(loaded_dict)
            else:
                pretrained_clip.load_state_dict(loaded_dict, strict=False)
                
            vt_state_dict = create_llava_vision_tower_dict(pretrained_clip)
            self.vision_tower.load_state_dict(vt_state_dict)
            
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    #@torch.no_grad() # I changed them
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    #@torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    #@torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
    
    

