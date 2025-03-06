import time
import math
from argparse import Namespace
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from clip_ft.openai_clip import CLIP_OPENAI
from clip_ft.utils import get_autocast
from clip_ft.lora_utils import LoRAParametrization, MaskedLoRAParametrization

from llava.train.train_utils import TrainingArguments


class ClipLoss(nn.Module):
    def __init__(
            self,
            cache_labels: bool = True,
    ):
        super().__init__()
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss
    
    
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
            
@torch.no_grad()
def mask_model_gradients(model: CLIP_OPENAI, mask: Dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        if param.requires_grad and any(nm in name for nm in ["mlp.fc1.weight", "mlp.c_fc.weight"]):
            assert param.grad is not None, f"The gradient of parameter `{name}` is None!"
            param.grad *= mask[name]     
      
            
def build_lorsu_mask(model: CLIP_OPENAI, 
                     dloader: DataLoader, 
                     training_args: TrainingArguments) -> Tuple[float, float, Dict[str, torch.Tensor], int]:
    
    time_scores, importance_scores = compute_importance_scores(model, dloader, training_args.grad_total_points_mask)
    mask_fc1 = {}
    tic = time.time()
    for name, param in model.named_parameters():
        if param.requires_grad and any(nm in name for nm in ["mlp.fc1.weight", "mlp.c_fc.weight"]):     
            magnitudes = importance_scores[name].abs()
            top_k_grads = int(magnitudes.numel() * training_args.sparsity)
            _, topk_indices = torch.topk(magnitudes.view(-1), k=top_k_grads)
            mask_fc1[name] = torch.zeros_like(magnitudes)
            mask_fc1[name].view(-1)[topk_indices] = 1.
    
    parametrize.register_parametrization(model.visual, 
                                        "proj", 
                                        LoRAParametrization(in_features=model.visual.proj.size(1), 
                                                            out_features=model.visual.proj.size(0),
                                                            rank=training_args.lorsu_rank, 
                                                            lora_alpha=training_args.lorsu_alpha).to(training_args.device)     
    )               
                                
    all_head_dim = model.visual.transformer.resblocks[0].attn.embed_dim
    num_heads = model.visual.transformer.resblocks[0].attn.num_heads
    attn_trainable = 0
    head_params = training_args.lorsu_rank * (model.visual.proj.size(0) + model.visual.proj.size(1))
    for i in range(len(model.visual.transformer.resblocks)): 
        ##############################
        # Multihead Self-Attention ###
        ##############################
        if training_args.top_k_heads > 0:
            dqkv = importance_scores[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"].reshape(3, num_heads, all_head_dim // num_heads, all_head_dim)
            score_attn = dqkv.square().sum(dim=(0, 2, 3))
            _, topk_heads = torch.topk(score_attn, k=training_args.top_k_heads)  
                
            mask_msa_qkv = torch.zeros_like(dqkv)
            mask_msa_qkv[:, topk_heads] = 1.
            mask_msa_qkv = mask_msa_qkv.reshape(3 * all_head_dim, all_head_dim)
            
            attn_layer = model.visual.transformer.resblocks[i].attn
            in_proj_outfeatures, in_proj_infeatures = attn_layer.in_proj_weight.shape
            params_attn = min(mask_msa_qkv.sum().item(), training_args.lorsu_rank * (in_proj_infeatures + in_proj_outfeatures))
            attn_trainable += params_attn
            
            parametrize.register_parametrization(attn_layer, "in_proj_weight", MaskedLoRAParametrization(in_features=in_proj_infeatures, 
                                                                                                            out_features=in_proj_outfeatures,
                                                                                                            mask=mask_msa_qkv, 
                                                                                                            rank=training_args.lorsu_rank, 
                                                                                                            lora_alpha=training_args.lorsu_alpha).to(training_args.device)  
            )
    
    model.requires_grad_(False)    
    trainable_param_names = ["c_fc", "lora"]
    if training_args.top_k_heads > 0:
        trainable_param_names.extend(["in_proj_bias"])
    for name, param in model.visual.named_parameters():
        param.requires_grad = any(trainable_p_name in name for trainable_p_name in trainable_param_names)
        if param.requires_grad:
            param.data = param.data.float()
                
    toc = time.time() - tic
    count_real_trainable_params = sum(int(item.sum().item()) for _, item in mask_fc1.items()) + head_params + attn_trainable
    return toc, time_scores, mask_fc1, count_real_trainable_params


def compute_importance_scores(model: CLIP_OPENAI, 
                             dloader: DataLoader, 
                             num_samples_grad: int) -> Tuple[float, Dict[str, torch.Tensor]]:
    input_dtype = model.dtype
    autocast = get_autocast(input_dtype)
    device = model.device        
    loss_f = ClipLoss()         
    importance_scores = {name: torch.zeros_like(param.data) for name, param in model.named_parameters() if param.requires_grad}

    tic = time.time()
    num_samples_used = 0
    for images, labels_or_texts in dloader:
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        labels_or_texts = labels_or_texts.to(device=device, non_blocking=True)
        num_samples_used += images.size(0)

        model.zero_grad()
        with autocast():
            image_features, text_features, logit_scale = model(images, labels_or_texts)
            total_loss = loss_f(image_features, text_features, logit_scale)
 
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, training stopped!")
        
        total_loss.backward()
        
        # Accumulate importance scores
        stop_flag = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                importance_scores[name] += (images.size(0) * param.grad.data.clone())
                if importance_scores[name].abs().min() < 1e-10:
                    stop_flag = False
                        
        if num_samples_used >= num_samples_grad and stop_flag:
            break
                           
    unbiased_scale = 1. / num_samples_used
    for name, param in model.named_parameters():
        if param.requires_grad:
            importance_scores[name] *= unbiased_scale         
    toc = time.time() - tic
    
    return toc, importance_scores




