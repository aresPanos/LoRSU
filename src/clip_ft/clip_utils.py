import json
import re
import os
import gc
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, List, Union, Tuple
from argparse import Namespace
from loguru._logger import Logger

import torch
from torch import nn

from open_clip import CLIPVisionCfg, CLIPTextCfg, \
                      convert_weights_to_fp16, get_pretrained_url, \
                      download_pretrained_from_url, list_openai_models, get_cast_dtype

from clip_ft.openai_clip import CLIP_OPENAI
from clip_ft.utils import get_input_dtype
from clip_ft.transform import PreprocessCfg
from clip_ft.lora_utils import add_lora_visual_openai, merge_lora
from clip_ft.transformer import Attention, TextTransformer

from llava.train.train_utils import TrainingArguments

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def convert_weights_to_lp(model: nn.Module, dtype: torch.dtype = torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)
    
def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu: bool = True,
        cast_dtype: torch.dtype = torch.float16,
        use_adalora: bool = False,
) -> Tuple[CLIP_OPENAI, CLIPVisionCfg]:
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP_OPENAI(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
        use_adalora=use_adalora,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    cfg = PreprocessCfg(image_size=image_size)
    return model.eval(), cfg


def load_openai_model(
        name: str,
        use_adalora: bool = False,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[str] = None,
) -> Tuple[CLIP_OPENAI, CLIPVisionCfg]:
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        state_dict = torch.load(model_path, map_location="cpu")

    # Build a non-jit model from the OpenAI jitted model state dict
    cast_dtype = get_cast_dtype(precision)
    try:
        model, cfg = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype, use_adalora=use_adalora)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model, cfg = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype, use_adalora=use_adalora)

    # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
    model = model.to(device)
    # FIXME support pure fp16/bf16 precision modes
    if precision != 'fp16':
        model.float()
        if precision == 'bf16':
            # for bf16, convert back to low-precision
            convert_weights_to_lp(model, dtype=torch.bfloat16)
    
    return model, cfg


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs()  # initial populate of model config registry

            

def load_clip_configs(cf: str):
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
        if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
            return model_cfg
        else:
            raise ValueError("Invalid path for EVA_CLIP_g_14.json configuration file.")
        

def list_models() -> List[str]:
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None
    

def load_state_dict_plus(checkpoint_path: str, 
                         checkpoint_fine_tuned_path: Optional[str] = None,
                         alpha: float = 0.5,
                         map_location: str='cpu', 
                         model_key: str = 'model|module|state_dict') -> OrderedDict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    for mk in model_key.split('|'):
        if isinstance(checkpoint, dict) and mk in checkpoint:
            state_dict = checkpoint[mk]
            break
        else:
            state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    if checkpoint_fine_tuned_path:
        checkpoint_subset = torch.load(checkpoint_fine_tuned_path, map_location=map_location)["state_dict"]
        for name, value in checkpoint_subset.items():
                dtype = state_dict[name].dtype
                assert state_dict[name].shape == value.shape, f"Parameter `{name}` should have shape {state_dict[name].shape} but {value.shape} was given."
                #state_dict[name] = alpha * value.to(dtype=dtype) + (1. - alpha) * state_dict[name]
                state_dict[name] = value.to(dtype=dtype)
                
    return state_dict



def create_model(training_args: TrainingArguments, lggr: Optional[Logger] = None) -> Tuple[CLIP_OPENAI, int, CLIPVisionCfg]:
    model, visual_cfg = load_openai_model(name='ViT-L-14-336', device='cpu')
    if training_args.ft_method == 'lora_v_clip':
        add_lora_visual_openai(model, training_args.lora_rank, training_args.lora_alpha)
  
    precision = get_input_dtype(training_args.precision)  
    if precision == torch.float16:
        assert training_args.device.type != 'cpu', "device must be `cuda` for precision fp16"
    convert_weights_to_lp(model, precision)
    model.to(device=training_args.device)
    print_fn = lggr.info if lggr else print
        
    if training_args.ft_method == 'lora_v_clip':
        model.freeze()
        for name, param in model.named_parameters():
            param.requires_grad = "lora" in name 
    else:
        model.freeze(training_args.ft_method)
    
    count_all_params = sum(param.numel() for name, param in model.visual.named_parameters() if not "lora" in name)
    if training_args.ft_method == 'lorsu_v_clip': 
        count_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        str_print = f"(OpenAI-CLIP-ViT-L-14-336) #parameters: {count_all_params},  #trainable params: {count_trainable_params}, percentage: {100 * count_trainable_params / count_all_params:.2f}%\n"
        print_fn(str_print)
            
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_fn(f"{name}: {param.shape}")

    return model, count_all_params, visual_cfg
        

def merge_model_ssl(model: CLIP_OPENAI, training_args: TrainingArguments) -> CLIP_OPENAI:
    merge_lora(model)
    model_updated, _, _ = create_model(training_args) 
    model_updated.load_state_dict(model.state_dict())
        
    return model_updated
    
    
def load_state_dict_general(checkpoint_path: str, 
                            map_location: Union[str, torch.device] = 'cpu', 
                            model_key: str = 'model|module|state_dict') -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    for mk in model_key.split('|'):
        if isinstance(checkpoint, dict) and mk in checkpoint:
            state_dict = checkpoint[mk]
            break
        else:
            state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
                
    return state_dict
   
def load_ckpt_ssl(training_args: TrainingArguments, fine_tuned_model_path: str) -> CLIP_OPENAI:
    model, _, _ = create_model(training_args)        
    state_dict = load_state_dict_general(fine_tuned_model_path, model.device)    
    model.visual.load_state_dict(state_dict)
        
    return model


@torch.no_grad()  
def state_dict_trainable(model: CLIP_OPENAI) -> OrderedDict:
    state_dictionary = OrderedDict()                       
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dictionary[name] = param
    return state_dictionary


@torch.no_grad()  
def state_dict_merged_ssl(model: CLIP_OPENAI) -> OrderedDict:
    merge_lora(model)
    return model.visual.state_dict()
                    

def save_model_ckpt(training_args: TrainingArguments, model: CLIP_OPENAI, epoch: int):
    st_dict = state_dict_merged_ssl(model) if training_args.ft_method == 'lorsu_v_clip' else state_dict_trainable(model)
        
    checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": st_dict,
                }
        
    torch.save(checkpoint_dict, training_args.ckpt_fname.format(epoch))
            
            
def save_model_ckpt_cl(training_args: TrainingArguments, model: CLIP_OPENAI, session: int, epoch: int):
    st_dict = state_dict_merged_ssl(model) if training_args.ft_method == 'lorsu_v_clip' else state_dict_trainable(model)
        
    checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": st_dict
    }
        
    torch.save(checkpoint_dict, training_args.ckpt_fname.format(session+1, epoch))
