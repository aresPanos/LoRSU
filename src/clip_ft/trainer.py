import math
import os
import time
from typing import Optional
from loguru._logger import Logger

from llava.train.train_utils import ModelArguments, DataArguments, TrainingArguments

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from transformers.trainer_pt_utils import get_model_param_count

from clip_ft.clip_utils import create_model, save_model_ckpt, save_model_ckpt_cl, load_ckpt_ssl
from clip_ft.data_utils import (get_cl_dataloaders, get_toyotahome_dl, get_vsr_dl,
                       get_gtsrb_dl, get_aircraft_dl, get_counteranimal_dl, 
                       get_eurosat_dl, get_hm_dl)
from clip_ft.utils import get_autocast, AverageMeter
from clip_ft.train_utils import ClipLoss, CosineAnnealingWarmupRestarts, mask_model_gradients, build_lorsu_mask
from clip_ft.openai_clip import CLIP_OPENAI

from llava.model.builder import load_pretrained_model
from llava.eval.eval_utils import set_seed
from llava.utils import disable_torch_init
from llava.train.train_utils import calculate_model_size_GB, print_trainable_parameters
from llava.model.builder import load_pretrained_model
from llava.eval.eval_vqa_utils import (eval_vsr, eval_hm, eval_dallevqa, eval_gtsrbvqa,
                                      eval_aircraftvqa, eval_counteranimalvqa, eval_tsivqa,
                                      eval_eurosatvqa, eval_mmvpvqa, eval_visonlyqa)

      
class CLIP_Trainer:
    def __init__(self, 
                 model_args: ModelArguments, 
                 data_args: DataArguments,
                 training_args: TrainingArguments,
                 lggr: Logger):        
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.lggr = lggr
        self.loss_meter = AverageMeter()
        self.lr_meter = AverageMeter()
        self.clip_loss = ClipLoss()
        
    def prepare_model_for_training(self, model: CLIP_OPENAI):
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
                
    def train_eval_main(self):
        model, count_all_params, cfg = create_model(self.training_args, self.lggr)  
        self.lggr.info(f"Size of the model: {calculate_model_size_GB(model):.2f} GB")
        if self.data_args.dataset == "tsi":
            train_dataloader = get_toyotahome_dl(self.data_args, train=True, labeled=False, cfg=cfg) 
        elif self.data_args.dataset == "vsr":    
            train_dataloader = get_vsr_dl(self.data_args, cfg)
        elif self.data_args.dataset == "hm":    
            train_dataloader = get_hm_dl(self.data_args, cfg)
        elif self.data_args.dataset == "aircraft":
            train_dataloader = get_aircraft_dl(self.data_args, train=True, labeled=False, cfg=cfg) 
        elif self.data_args.dataset == "eurosat":
            train_dataloader = get_eurosat_dl(self.data_args, train=True, labeled=False, cfg=cfg) 
        elif self.data_args.dataset == "counteranimal":
            train_dataloader = get_counteranimal_dl(self.data_args, train=True, labeled=False, cfg=cfg) 
        else:    
            train_dataloader = get_gtsrb_dl(self.data_args, train=True, labeled=False, cfg=cfg)

        num_update_steps_per_epoch = len(train_dataloader)
        num_examples = len(train_dataloader.dataset)
        max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(self.training_args.num_train_epochs)

        self.lggr.info("***** Running training *****")
        self.lggr.info(f"  Num examples = {num_examples:,}")
        self.lggr.info(f"  Num Epochs = {num_train_epochs:,}")
        self.lggr.info(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size:,}")
        self.lggr.info(f"  Gradient Accumulation steps = {self.training_args.gradient_accumulation_steps}")
        self.lggr.info(f"  Total optimization steps = {max_steps:,}")
        
        if self.training_args.ft_method == "lorsu_v_clip":
            time_mask, time_scores, self.grad_mask_lorsu, count_real_trainable_params = build_lorsu_mask(model, train_dataloader, self.training_args)
            self.lggr.info(f"LoRSU mask was built. Mask time={time_mask / 60. :.1f} mins, Score time={time_scores / 60. :.1f} mins")
            self.lggr.info(f"(OpenAI-CLIP-ViT-L-14-336) #parameters: {count_all_params},  #learned params: {count_real_trainable_params}, percentage: {100 * count_real_trainable_params / count_all_params:.2f}%\n")
         
        self.prepare_model_for_training(model)
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.training_args.learning_rate)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                     first_cycle_steps=self.training_args.num_train_epochs * len(train_dataloader),
                                                     cycle_mult=1.0,
                                                     max_lr=self.training_args.learning_rate,
                                                     min_lr=1e-6,
                                                     warmup_steps=1)

        scaler = GradScaler()
        print_trainable_parameters(model, self.lggr)
        for epoch in range(self.training_args.num_train_epochs):
            self.train_one_epoch(model, train_dataloader, epoch, optimizer, lr_scheduler, scaler)
            completed_epoch = epoch + 1   
            
        save_model_ckpt(self.training_args, model, completed_epoch, optimizer, scaler)
        
    def train_eval_cl(self):
        model, count_all_params, cfg = create_model(self.training_args, self.lggr)       
        list_train_dl  = get_cl_dataloaders(self.data_args, self.training_args, cfg)
        self.lggr.info(f"Size of the initial model: {calculate_model_size_GB(model):.2f} GB")
        for session in range(self.data_args.num_sessions):
            self.lggr.info(f"#################   Start of Session {session+1}   #################")
            train_dataloader = list_train_dl[session]
            
            num_update_steps_per_epoch = len(train_dataloader)
            num_examples = len(train_dataloader.dataset)
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.training_args.num_train_epochs)

            self.lggr.info("***** Running training *****")
            self.lggr.info(f"  Num examples = {num_examples:,}")
            self.lggr.info(f"  Num Epochs = {num_train_epochs:,}")
            self.lggr.info(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size:,}")
            self.lggr.info(f"  Total optimization steps = {max_steps:,}")
            
            if self.training_args.ft_method == "lorsu_v_clip":
                time_mask, time_scores, self.grad_mask_lorsu, count_real_trainable_params = build_lorsu_mask(model, train_dataloader, self.training_args)
                self.lggr.info(f"LoRSU mask was built. Mask time={time_mask / 60. :.1f} mins, Score time={time_scores / 60. :.1f} mins")
                self.lggr.info(f"(OpenAI-CLIP-ViT-L-14-336) #parameters: {count_all_params},  #learned params: {count_real_trainable_params}, percentage: {100 * count_real_trainable_params / count_all_params:.2f}%\n")

                
            self.prepare_model_for_training(model)
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=self.training_args.learning_rate)
            lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                         first_cycle_steps=self.training_args.num_train_epochs * len(train_dataloader),
                                                         cycle_mult=1.0,
                                                         max_lr=self.training_args.learning_rate,
                                                         min_lr=1e-6,
                                                         warmup_steps=1)
            scaler = GradScaler()
            print_trainable_parameters(model, self.lggr)
            for epoch in range(num_train_epochs):
                self.train_one_epoch(model, train_dataloader, epoch, optimizer, lr_scheduler, scaler)
                completed_epoch = epoch + 1    
            
            save_model_ckpt_cl(self.training_args, model, session, completed_epoch)
            if self.training_args.ft_method == 'lorsu_v_clip': 
                model = load_ckpt_ssl(self.training_args, self.training_args.ckpt_fname.format(session+1, completed_epoch))                  
            self.lggr.info(f"#################   End of Session {session + 1}   #################\n\n")  
            
        save_model_ckpt_cl(self.training_args, model, completed_epoch, session)
    
    def train_eval(self):
        if self.data_args.is_cl:
            self.train_eval_cl()
        else:
            self.train_eval_main()
        self.eval_vqa()
                                              
    def train_one_epoch(self, 
                        model: CLIP_OPENAI, 
                        dloader: DataLoader, 
                        epoch: int,
                        optimizer: optim.Optimizer, 
                        scheduler: Optional[_LRScheduler] = None, 
                        scaler: Optional[GradScaler] = None):
        
        input_dtype = model.dtype
        autocast = get_autocast(input_dtype)
        device = model.device
        
        model.train()
        self.loss_meter.reset()
        self.lr_meter.reset()
        tic = time.time()
        for i_batch, (images, texts) in enumerate(dloader):
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            bsize = images.size(0)
            
            optimizer.zero_grad()
            with autocast():
                image_features, text_features, logit_scale = model(images, texts)
                total_loss = self.clip_loss(image_features, text_features, logit_scale)               
                
            loss_value = total_loss.item()
            if not math.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, training stopped!")
  
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if self.training_args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip_norm, norm_type=2.0)
                if self.training_args.ft_method == 'lorsu_v_clip':
                    mask_model_gradients(model, self.grad_mask_lorsu)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if self.training_args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip_norm, norm_type=2.0)  
                if self.training_args.ft_method == 'lorsu_v_clip':
                    mask_model_gradients(model, self.grad_mask_lorsu)  
                optimizer.step()
                
            self.loss_meter.update(loss_value, bsize)
            self.lr_meter.update(optimizer.param_groups[0]["lr"])      
            
        if scheduler:
            scheduler.step()  
        toc = time.time() - tic
        if torch.cuda.is_available():
            gb = 1024.0 ** 3
            gpu_memory = torch.cuda.max_memory_allocated() / gb
            self.lggr.info(f"(train) Epoch [{epoch+1}/{math.ceil(self.training_args.num_train_epochs)}]: loss={self.loss_meter.avg:.4f}, lr={self.lr_meter.avg:.6f}, Elapsed time={toc / 60.0:.1f} mins, GPU memory allocated={gpu_memory:.1f} GB")
        else:
            self.lggr.info(f"(train) Epoch [{epoch+1}/{math.ceil(self.training_args.num_train_epochs)}]: loss={self.loss_meter.avg:.4f}, lr={self.lr_meter.avg:.6f}, Elapsed time={toc / 60.0:.1f} mins")
            
    def eval_vqa(self):
        set_seed(self.training_args.seed)
    
        disable_torch_init()
        if self.data_args.is_cl:
            fine_tuned_clip_model = self.training_args.ckpt_fname.format(self.data_args.num_sessions, math.ceil(self.training_args.num_train_epochs))
        else:
            fine_tuned_clip_model = self.training_args.ckpt_fname.format(math.ceil(self.training_args.num_train_epochs))
            
        tokenizer, model, image_processor, context_len = load_pretrained_model(self.model_args.model_path, 
                                                                               self.model_args.model_base, 
                                                                               self.model_args.model_name, 
                                                                               fine_tuned_clip_model)
        model.eval()    
        gb = 1024.0 ** 3
        
        acc_gtsrb, num_samples_gtsrb, time_elapsed_gtsrb = eval_gtsrbvqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"GTSRB: acc={acc_gtsrb:.2f}%\t#Samples={num_samples_gtsrb}\tET={time_elapsed_gtsrb/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")

        acc_tsivqa, num_samples_tsivqa, time_elapsed_tsivqa = eval_tsivqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"TSIVQA: acc={acc_tsivqa:.2f}%\t#Samples={num_samples_tsivqa}\tET={time_elapsed_tsivqa/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_counteranimal, num_samples_counteranimal, time_elapsed_counteranimal = eval_counteranimalvqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"CounterAnimal: acc={acc_counteranimal:.2f}%\t#Samples={num_samples_counteranimal}\tET={time_elapsed_counteranimal/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
                
        acc_aircraft, num_samples_aircraft, time_elapsed_aircraft = eval_aircraftvqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"FGVC-Aircraft: acc={acc_aircraft:.2f}%\t#Samples={num_samples_aircraft}\tET={time_elapsed_aircraft/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_eurosat, num_samples_eurosat, time_elapsed_eurosat = eval_eurosatvqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"EuroSAT: acc={acc_eurosat:.2f}%\t#Samples={num_samples_eurosat}\tET={time_elapsed_eurosat/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_dalle, num_samples_dalle, time_elapsed_dalle = eval_dallevqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"DALLE-VQA: acc={acc_dalle:.2f}%\t#Samples={num_samples_dalle}\tET={time_elapsed_dalle/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_vsr, num_samples_vsr, time_elapsed_vsr = eval_vsr(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"VSR: acc={acc_vsr:.2f}%\t#Samples={num_samples_vsr}\tET={time_elapsed_vsr/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
            
        acc_hm, num_samples_hm, time_elapsed_hm = eval_hm(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"HM: acc={acc_hm:.2f}%\t#Samples={num_samples_hm}\tET={time_elapsed_hm/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_mmvp, num_samples_mmvp, time_elapsed_mmvp = eval_mmvpvqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"MMVP-VQA: acc={acc_mmvp:.2f}%\t#Samples={num_samples_mmvp}\tET={time_elapsed_mmvp/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_visonlyqa, num_samples_visonlyqa, time_elapsed_visonlyqa = eval_visonlyqa(self.model_args, self.data_args, self.training_args, model, image_processor, tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"VisOnlyQA: acc={acc_visonlyqa:.2f}%\t#Samples={num_samples_visonlyqa}\tET={time_elapsed_visonlyqa/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")  
            
        