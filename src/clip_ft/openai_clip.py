from dataclasses import dataclass
from typing import Tuple, Union, List, Optional

import numpy as np

import torch

from open_clip import CLIP

def disabled_train(self, mode: bool=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models
       
    
class CLIP_OPENAI(CLIP):
    def __init__(self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            use_adalora: bool = False,
    ):
        super().__init__(embed_dim, vision_cfg, text_cfg, quick_gelu, init_logit_scale, init_logit_bias, cast_dtype, output_dict, use_adalora)
        self.requires_grad_(False)
        
    @property
    def dtype(self):
        return self.transformer.resblocks[-1].mlp.c_fc.weight.dtype
    
    @property
    def device(self) -> torch.device:
        return self.logit_scale.device    
               
    def freeze_encoder(self, is_visual: bool = True, list_names: List[str] = [""], freeze_bias: bool = False):
        """
        Freeze all the parameters of the visual ot text encoder except those with names in list_names.
        """
        encoder = self.visual if is_visual else self.transformer
        for name, param in encoder.named_parameters():
            param.requires_grad = any(edit_layer in name for edit_layer in list_names)
            if freeze_bias:
                param.requires_grad = param.requires_grad and ("bias" not in name)
            if param.requires_grad:
                 param.data = param.data.float()
                 
        if len(list_names) == 0:
            encoder = encoder.eval()
            #encoder.train = disabled_train
            
    def freeze(self, mode_visual: str = "all"):
        assert mode_visual in ["layer_norm", "lorsu_v_clip", "all", "full_ft"], f"The `mode_visual` argument should have the value of [`layer_norm`, `spu`, `all`, `ssl`] but {mode_visual} was given"
         
        if mode_visual == "layer_norm":
            self.freeze_encoder(list_names=["ln"])
        elif mode_visual == "lorsu_v_clip":
            self.freeze_encoder(list_names=["attn.in_proj_weight", "mlp.c_fc"]) 
            self.visual.proj.requires_grad_(True)
        elif mode_visual == "all":
            self.freeze_encoder(list_names=[])
        else:
            self.freeze_encoder() 
            
        self.freeze_encoder(False, list_names=[])
    
