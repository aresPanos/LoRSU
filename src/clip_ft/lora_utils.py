import math
from functools import partial
from typing import Dict, List, Optional, Callable, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.parametrize as parametrize

from clip_ft.openai_clip import CLIP_OPENAI


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
        

class LoRALinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

# ---------------- LoRA --------------------------------

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
    

    
    
class MaskedLoRAParametrization(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 mask: torch.Tensor, 
                 rank: int = 4, 
                 lora_alpha: float = 1.0):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        assert rank > 0, f"Rank should be a positive integer but rank={rank} was given."
        assert mask.size(0) == out_features and mask.size(1) == in_features, f"`mask` should be a tensor of shape [{out_features}, {in_features}] but a tensor of shape {mask.shape} was given."
        self.lora_A = nn.Parameter(torch.zeros((out_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, in_features)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        scaling = lora_alpha / rank
        self.scaled_mask = scaling * mask
        self.forward_fn = self.lora_forward

    def lora_forward(self, X: torch.Tensor):
        return X + self.scaled_mask * (self.lora_A@self.lora_B)

    def forward(self, X: torch.Tensor):
        return self.forward_fn(X)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self.lora_forward
        
        
class MaskedLoRALinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_lora: torch.Tensor,
        r: int = 0,
        lora_alpha: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert mask_lora.size(0) == out_features and mask_lora.size(1) == in_features, f"`mask` should be a tensor of shape [{out_features}, {in_features}] but a tensor of shape {mask_lora.shape} was given."
        
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.mask_lora = nn.Parameter(mask_lora, requires_grad=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.mask_lora.float() * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.mask_lora.float() * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.mask_lora.float() * self.scaling
                )
                self.merged = False
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            merged_masked_weight = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.mask_lora.float() * self.scaling
            return F.linear(x, transpose(merged_masked_weight, self.fan_in_fan_out), bias=self.bias)
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        

def get_lora_config(rank: int = 4, lora_alpha: float = 1.0) -> Dict[nn.Linear,  Dict[str, LoRAParametrization]]:
    lora_config = {
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha),
        },
    }
    return lora_config

def apply_lora(layer: nn.Module, 
               rank: int = 2, 
               lora_alpha: float = 1.0, 
               register: bool = True, 
               merge: bool = False):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        lora_config = get_lora_config(rank, lora_alpha)
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)
                
                

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
                              
                

def add_lora(model: nn.Module, rank: int, lora_alpha: float = 1.0):
    """add lora parametrization to all layers in a model. Calling it twice will add lora twice"""
    model.apply(partial(apply_lora, rank=rank, lora_alpha=lora_alpha))
    
    
def add_lora_openai(model: nn.Module, rank: int, lora_alpha: float = 1.0):
    """add lora parametrization to all layers in a model. Calling it twice will add lora twice"""
    model.apply(partial(apply_lora_openai, rank=rank, lora_alpha=lora_alpha))
            
    
def add_lora_visual_openai(model: CLIP_OPENAI, rank: int, lora_alpha: float = 1.0):
    assert isinstance(model, CLIP_OPENAI), f"The model should an instance of class CLIP_OPENAI but {type(model)} was given."
    """add LoRA parametrization to all layers in the visual encoder of CLIP. Calling it twice will add lora twice"""
    add_lora_openai(model.visual, rank, lora_alpha)
    
    out_features, in_features = model.visual.proj.shape
    parametrize.register_parametrization(model.visual, "proj", LoRAParametrization(in_features, out_features, rank, lora_alpha).to(device=model.visual.proj.device))
      

def add_lora_by_name(model: nn.Module, target_module_names: List[str], lora_config: Dict[nn.Linear,  Dict[str, LoRAParametrization]]):
    """Add LoRA parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_lora(layer, lora_config=lora_config)
            

def merge_lora(model: CLIP_OPENAI):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=True))
       
    

def remove_lora(model: CLIP_OPENAI):
    """remove lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=False))
    

def apply_to_lora(fn):
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn


def enable_lora(model: CLIP_OPENAI):
    model.apply(apply_to_lora(lambda x: x.enable_lora()))
    
    
def disable_lora(model: CLIP_OPENAI):
    model.apply(apply_to_lora(lambda x: x.disable_lora()))
    
    
def get_submodules(model: nn.Module, key: str):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name
    
    
def replace_module(parent_module, child_name, new_module: Union[nn.Linear, MaskedLoRALinear], old_module: Union[nn.Linear, MaskedLoRALinear]):
    setattr(parent_module, child_name, new_module)
    if isinstance(new_module, MaskedLoRALinear) or isinstance(new_module, LoRALinear):
        assert isinstance(old_module, nn.Linear)
        new_module.weight = old_module.weight
    elif isinstance(new_module, nn.Linear):
        assert isinstance(old_module, MaskedLoRALinear) or isinstance(old_module, LoRALinear)
        if isinstance(old_module, MaskedLoRALinear):
            new_module.weight.data = old_module.weight.data + transpose(old_module.lora_B.weight.data @ old_module.lora_A.weight.data, old_module.fan_in_fan_out) * old_module.mask_lora.float() * old_module.scaling  
        else:
            new_module.weight.data = old_module.weight.data + transpose(old_module.lora_B.weight.data @ old_module.lora_A.weight.data, old_module.fan_in_fan_out) * old_module.scaling
        new_module.weight.data = new_module.weight.data.type(old_module.weight.data.dtype)
    else:
        raise ValueError(f"old_module should be an instance of class `nn.Linear` or `MaskedLoRALinear` but {type(old_module).__name__} was given.")
    if old_module.bias is not None:
        new_module.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)    
                
            
def calculate_model_size_GB(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_gb = (param_size + buffer_size) / 1024**3
    return size_all_gb


# ------------------- helper function for collecting parameters for training/saving -------------------


def name_is_lora_visual(name: str) -> bool:
    return name.split(".")[-1] in ["lora_A", "lora_B"] and "visual" in name


def name_is_bias_visual(name: str) -> bool:
    return name.split(".")[-1] == "bias" and "visual" in name


def get_params_by_name(model: CLIP_OPENAI, print_shapes: bool=False, name_filter: Optional[Callable] = None):
    use_filter = name_filter is not None
    for n, p in model.named_parameters():
        if use_filter:
            if name_filter(n):
                if print_shapes:
                    print(f"{n}: {p.shape}")
                yield p
        else:
            if print_shapes:
                print(f"{n}: {p.shape}")
            yield p


def get_lora_params(model: CLIP_OPENAI, print_shapes: bool = False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_lora_visual)


def get_bias_params(model: CLIP_OPENAI, print_shapes: bool = False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_bias_visual)


def get_lora_state_dict(model: CLIP_OPENAI) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if name_is_lora_visual(k)}
