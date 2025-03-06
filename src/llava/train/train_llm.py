import math
from argparse import Namespace
from typing import Optional, Union, Any, Callable, List, Dict
from loguru._logger import Logger
import inspect
from packaging import version
import time
import datetime
from dataclasses import asdict
import gc

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import find_labels
from transformers.trainer_utils import seed_worker, RemoveColumnsCollator
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer import get_parameter_names, has_length, ALL_LAYERNORM_LAYERS

from peft import PeftModel

import datasets

from llava.model import LlavaLlamaForCausalLM
from llava.train.llava_trainer import  LengthGroupedSampler
from llava.train.train_utils import (ModelArguments, DataArguments, TrainingArguments,
                                    MetricLoggerLoguru, SmoothedValue, LinearWarmupCosineLRScheduler,
                                    calculate_model_size_GB, prepare_inputs, print_trainable_parameters,
                                    make_supervised_data_module, build_spu_attn_mask, merge_weights, mask_model_gradients,
                                    mask_model_gradients_vision_tower)
                                    
from llava.eval.eval_vqa_utils import (eval_vsr, eval_hm, eval_dallevqa, eval_gtsrbvqa, 
                                      eval_aircraftvqa, eval_counteranimalvqa, eval_tsivqa, 
                                      eval_eurosatvqa, eval_mmvpvqa, eval_visonlyqa)


class LLaVA_Trainer:
    def __init__(
        self,
        model: Union[LlavaLlamaForCausalLM, PeftModel],
        model_args: ModelArguments, 
        data_args: DataArguments,
        training_args: TrainingArguments,
        tokenizer: PreTrainedTokenizer,
        lggr: Logger,
    ):
        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )
            
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.lggr = lggr
        
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.training_args.label_names is None else self.training_args.label_names
        self._train_batch_size = self.training_args.train_batch_size
        
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        
    def restart_ftp(self):
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._lr_sched = None
        self.start_epoch = 0
        self.current_session = 0        
            
    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.training_args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.training_args.train_batch_size,
                world_size=self.training_args.world_size * self.training_args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return RandomSampler(train_dataset)
        
    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            self.lggr.warning("Cannot get num_tokens from dataloader")
            return train_tokens
        
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if isinstance(self.model, PeftModel):
                model_to_inspect = self.model.get_base_model()
                
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
        
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.training_args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            self.lggr.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
        
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.training_args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=self.lggr,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator
        
    def get_train_dataloader(self, classes_session: Optional[Union[List[int], List[str]]]= None) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        train_dataset, data_collator = make_supervised_data_module(tokenizer=self.tokenizer,  
                                                                   data_args=self.data_args,
                                                                   classes_session=classes_session)
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.training_args.dataloader_num_workers,
            "pin_memory": self.training_args.dataloader_pin_memory,
            "persistent_workers": self.training_args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.training_args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)
    
    def get_train_dataloader_list(self)-> List[DataLoader]:      
        list_train_dl = []
        if self.data_args.dataset == "tsi":
            session_classes_list = [['WatchTV', 'Laydown', 'Sitdown', 'Pour.Fromkettle', 'Enter', 'Drink.Frombottle'],
                                        ['Eat.Attable', 'Pour.Frombottle', 'Cook.Cleandishes', 'Maketea.Boilwater', 'Leave', 'Cook.Cleanup'],
                                        ['Maketea.Insertteabag', 'Makecoffee.Pourwater', 'Drink.Fromcan', 'Readbook', 'Cutbread'],
                                        ['Drink.Fromcup', 'Drink.Fromglass', 'Usetablet', 'Pour.Fromcan', 'Usetelephone'],
                                        ['Walk', 'Cook.Stir', 'Makecoffee.Pourgrains', 'Cook.Cut', 'Uselaptop'],
                                    ]
        elif self.data_args.dataset == "vsr":
            session_classes_list = [['oven', 'dining table', 'spoon', 'boat', 'cake', 'donut', 'sandwich'],
                                    ['fire hydrant', 'elephant', 'airplane', 'truck', 'apple', 'hot dog', 'sheep'],
                                    ['kite', 'baseball glove', 'cow', 'tie', 'scissors', 'toaster', 'tv'],
                                    ['bicycle', 'banana', 'couch', 'teddy bear', 'bus', 'umbrella', 'bird'],
                                    ['potted plant', 'bowl', 'broccoli', 'bottle', 'knife', 'orange', 'person', 'pizza'],
                                   ]
        elif self.data_args.dataset == "aircraft":
            classes = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 
                    19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 
                    84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0,  72, 35, 
                    58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 
                    32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
            session_classes_list = np.array_split(classes, self.data_args.num_sessions)
        elif self.data_args.dataset == "eurosat":
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            session_classes_list = np.array_split(classes, self.data_args.num_sessions)
        elif self.data_args.dataset == "counteranimal":
            classes = [102, 9, 20, 56, 23, 30, 357, 291, 144, 
                    41, 293, 42, 49, 54, 57, 70, 279, 305, 
                    71, 10, 76, 79, 349, 16, 81, 83, 100,
                    130, 30, 133, 150, 275, 276, 58, 277, 80, 
                    39, 290, 37, 296, 316, 337, 89, 360, 128]
            session_classes_list = np.array_split(classes, self.data_args.num_sessions)
        elif self.data_args.dataset in ["mmvp", "visonlyqa", "hm"]:
            session_classes_list = np.arange(self.data_args.num_sessions)        
        else:
            classes = [25, 2, 11, 1, 40, 27, 5, 9, 17, 32, 29, 20, 39, 21, 15, 23, 10, 3, 18, 38,
                    42, 14, 22, 35, 34, 19, 33, 12, 26, 41, 0, 37, 6, 13, 24, 30, 28, 31, 7, 16, 4, 36, 8]
            session_classes_list = np.array_split(classes, self.data_args.num_sessions)
        
        for classes_session in session_classes_list:
            dl_train = self.get_train_dataloader(classes_session)
            list_train_dl.append(dl_train)
        return list_train_dl 

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            self.lggr.info("*** Trainable parameters: `name`: `shape` ***")
            num_trainable_parameters = 0
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.lggr.info(f"{n}: {p.shape} ({p.dtype})")
                    num_trainable_parameters += p.data.nelement()
            self.lggr.info("Number of trainable parameters (++): %d" % num_trainable_parameters)
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.training_args.mm_projector_lr:
                projector_parameters = [name for name, _ in self.model.named_parameters() if "mm_projector" in name and any(nn in name for nn in ["bias", "lora_"])]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.training_args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.training_args.weight_decay,
                        "lr": self.training_args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.training_args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.training_args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            #optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.training_args)
            #self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=float(self.training_args.learning_rate),
                weight_decay=float(self.training_args.weight_decay),
                betas=(0.9, 0.999),
            )
            
    
    """
    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        # ""
        #Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        #passed as an argument.

        #Args:
        #    num_training_steps (int): The number of training steps to do.
        # ""
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.training_args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.training_args.lr_scheduler_kwargs,
            )

        return self.lr_scheduler
    """
    def create_scheduler(self, iters_per_epoch: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            assert self.optimizer is not None, "Optimizer should be defined before initializing lr_scheduler"
            self.lr_scheduler = LinearWarmupCosineLRScheduler(optimizer=self.optimizer,
                                                              max_epoch=self.training_args.num_train_epochs,
                                                              iters_per_epoch=iters_per_epoch,
                                                              min_lr=self.training_args.min_lr,
                                                              init_lr=self.training_args.learning_rate,
                                                              decay_rate=self.training_args.lr_decay_rate,
                                                              warmup_start_lr=self.training_args.warmup_lr,
                                                              warmup_steps=iters_per_epoch,)

        return self.lr_scheduler
    
    def create_optimizer_scheduler_scaler(self, iters_per_epoch: int):            
        self.create_optimizer()        
        self.create_scheduler(iters_per_epoch=iters_per_epoch)
        self.scaler = torch.cuda.amp.GradScaler()
        
    def reset_optimizer_scheduler(self, iters_per_epoch: int):
        if self.current_session > 0:
            optim_dict = self.optimizer.state_dict()
            scaler_dict = self.scaler.state_dict()
        
            del self.lr_scheduler, self.optimizer, self.scaler
            gc.collect()
            torch.cuda.empty_cache()
            self.lr_scheduler, self.optimizer, self.scaler = None, None, None
        
        self.create_optimizer_scheduler_scaler(iters_per_epoch)
        if self.current_session > 0:
            self.optimizer.load_state_dict(optim_dict)
            self.scaler.load_state_dict(scaler_dict)
        
    def prepare_model_for_training(self):
        self.model.zero_grad()                
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        
    def train_eval_offline(self):
        trainloader = self.get_train_dataloader()
        
        num_update_steps_per_epoch = len(trainloader)
        num_examples = len(trainloader.dataset)
        max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(self.training_args.num_train_epochs)
        num_trainable_params = get_model_param_count(self.model, trainable_only=True)
        percent_trainable = 100 * num_trainable_params / self.model_args.num_params_init
        
        self.lggr.info("***** Running training *****")
        self.lggr.info(f"  Num examples = {num_examples:,}")
        self.lggr.info(f"  Num Epochs = {num_train_epochs:,}")
        self.lggr.info(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size:,}")
        self.lggr.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self._train_batch_size:,}")
        self.lggr.info(f"  Gradient Accumulation steps = {self.training_args.gradient_accumulation_steps}")
        self.lggr.info(f"  Total optimization steps = {max_steps:,}")
        if 'lorsu' in self.training_args.ft_method:                   
            count_all_params = sum(param.numel() for param in self.model.parameters())
            self.grad_mask_lorsu, time_mask, time_scores, num_trainable_params = build_spu_attn_mask(self.model, trainloader, self.training_args)
            self.lggr.info(f"SPU-Attn mask was built. Mask time={time_mask / 60. :.1f} mins, Score time={time_scores / 60. :.1f} mins")
            
        self.lggr.info(f"  Number of parameters = {count_all_params:,}")
        self.lggr.info(f"  Number of trainable parameters = {num_trainable_params:,}")
        self.lggr.info(f"  Percentage of trainable parameters: {percent_trainable:.2f}%")
        self.lggr.info(f"  Size of the model: {calculate_model_size_GB(self.model):.2f} GB")

        print_trainable_parameters(self.model, self.lggr)

        self.prepare_model_for_training()
        self.create_optimizer_scheduler_scaler(iters_per_epoch=num_update_steps_per_epoch)
        start_time = time.time()
        for cur_epoch in range(num_train_epochs):
            # training phase
            self.lggr.info(f"Start training of epoch [{cur_epoch+1}]")
            train_stats = self.train_epoch(trainloader, cur_epoch)
            self.log_stats(train_stats)
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.lggr.info("Training time {}".format(total_time_str))
        self.eval_vqa_datasets()
            
    def train_eval_cl(self):
        trainloader_list = self.get_train_dataloader_list()
        self.current_session = 0
        start_time = time.time()
        sz_gb = calculate_model_size_GB(self.model)
        self.lggr.info(f"Size of the initial model: {sz_gb:.2f} GB")
        for session in range(self.data_args.num_sessions):
            self.lggr.info(f"#################   Start of Session {session+1}   #################")
            trainloader = trainloader_list[session]
            
            num_update_steps_per_epoch = len(trainloader)
            num_examples = len(trainloader.dataset)
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.training_args.num_train_epochs)
            num_trainable_params = get_model_param_count(self.model, trainable_only=True)
            percent_trainable = 100 * num_trainable_params / self.model_args.num_params_init
            
            self.lggr.info("***** Running training *****")
            self.lggr.info(f"  Num examples = {num_examples:,}")
            self.lggr.info(f"  Num Epochs = {num_train_epochs:,}")
            self.lggr.info(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size:,}")
            self.lggr.info(f"  Total optimization steps = {max_steps:,}")
            
            if 'lorsu' in self.training_args.ft_method:   
                if session > 0:
                    merge_weights(self.model, self.training_args)
                    sz_gb = calculate_model_size_GB(self.model)
                    self.lggr.info(f"Size of the merged model: {sz_gb:.2f} GB")
                    
                self.grad_mask_lorsu, time_mask, time_scores, num_trainable_params = build_spu_attn_mask(self.model, trainloader, self.training_args)
                self.lggr.info(f"SPU-Attn mask was built. Mask time={time_mask / 60. :.1f} mins, Score time={time_scores / 60. :.1f} mins")
           
            self.lggr.info(f"  Number of parameters = {self.model_args.num_params_init:,}")
            self.lggr.info(f"  Number of trainable parameters = {num_trainable_params:,}")
            self.lggr.info(f"  Percentage of trainable parameters: {percent_trainable:.2f}%")
            self.lggr.info(f"  Size of the model: {calculate_model_size_GB(self.model):.2f} GB")    
            print_trainable_parameters(self.model, self.lggr)
            
            self.prepare_model_for_training()
            self.reset_optimizer_scheduler(iters_per_epoch=num_update_steps_per_epoch)
            for cur_epoch in range(num_train_epochs):
                # training phase
                self.lggr.info(f"Start training of epoch [{cur_epoch+1}]")
                train_stats = self.train_epoch(trainloader, cur_epoch)
                self.log_stats(train_stats)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            self.lggr.info("Training time {} of Session {}".format(total_time_str, session+1))
            self.lggr.info(f"#################   End of Session {session + 1}   #################\n\n")
            self.current_session += 1
            
        self.eval_vqa_datasets()
        
    def train_eval_cl(self):
        trainloader_list = self.get_train_dataloader_list()
        self.current_session = 0
        start_time = time.time()
        sz_gb = calculate_model_size_GB(self.model)
        self.lggr.info(f"Size of the initial model: {sz_gb:.2f} GB")
        for session in range(self.data_args.num_sessions):
            self.lggr.info(f"#################   Start of Session {session+1}   #################")
            trainloader = trainloader_list[session]
            
            num_update_steps_per_epoch = len(trainloader)
            num_examples = len(trainloader.dataset)
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.training_args.num_train_epochs)
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.training_args.num_train_epochs)
            num_trainable_params = get_model_param_count(self.model, trainable_only=True)
            percent_trainable = 100 * num_trainable_params / self.model_args.num_params_init
            
            self.lggr.info("***** Running training *****")
            self.lggr.info(f"  Num examples = {num_examples:,}")
            self.lggr.info(f"  Num Epochs = {num_train_epochs:,}")
            self.lggr.info(f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size:,}")
            self.lggr.info(f"  Total optimization steps = {max_steps:,}")
            
            if 'lorsu' in self.training_args.ft_method:
                if session > 0:
                    merge_weights(self.model, self.training_args)
                    sz_gb = calculate_model_size_GB(self.model)
                    self.lggr.info(f"Size of the merged model: {sz_gb:.2f} GB")
                    
                count_all_params = sum(param.numel() for param in self.model.parameters())
                self.grad_mask_lorsu, time_mask, time_scores, count_real_trainable_params = build_spu_attn_mask(self.model, trainloader, self.training_args)
                self.lggr.info(f"SPU-Attn mask was built. Mask time={time_mask / 60. :.1f} mins, Score time={time_scores / 60. :.1f} mins")
                self.lggr.info(f"#parameters: {count_all_params},  #learned params: {count_real_trainable_params}, percentage: {100 * count_real_trainable_params / count_all_params:.2f}%\n")
                sz_gb = calculate_model_size_GB(self.model)
                self.lggr.info(f"Size of the modified model: {sz_gb:.2f} GB")
            else:
                self.lggr.info(f"  Number of trainable parameters = {num_trainable_params:,}")
                self.lggr.info(f"  Percentage of trainable parameters: {percent_trainable:.2f}%")
                self.lggr.info(f"  Size of the modified model: {calculate_model_size_GB(self.model):.2f} GB")
                
            print_trainable_parameters(self.model, self.lggr)
            self.prepare_model_for_training()
            self.reset_optimizer_scheduler(iters_per_epoch=num_update_steps_per_epoch)
            for cur_epoch in range(num_train_epochs):
                # training phase
                self.lggr.info(f"Start training of epoch [{cur_epoch+1}]")
                train_stats = self.train_epoch(trainloader, cur_epoch)
                self.log_stats(train_stats)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            self.lggr.info("Training time {} of Session {}".format(total_time_str, session+1))
            self.lggr.info(f"#################   End of Session {session + 1}   #################\n\n")
            self.current_session += 1
            
        self.eval_vqa_datasets()
                        
    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        # train
        self.model.train()
        iters_per_epoch = len(data_loader)
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLoggerLoguru(lggr=self.lggr, delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        self.lggr.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")
        header = "Train: data epoch: [{}]".format(epoch)

        for step_i in metric_logger.log_every(range(iters_per_epoch), self.training_args.log_freq, header):
            if step_i >= iters_per_epoch:
                break         
            
            inputs = next(data_loader)
            inputs = prepare_inputs(inputs, cuda_enabled=True)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=step_i)
            with torch.cuda.amp.autocast(enabled=True):
                loss = self.model(**inputs)["loss"]
                
            self.scaler.scale(loss).backward()
            if 'lorsu' in self.training_args.ft_method: 
                if self.training_args.ft_method == "lorsu_llm":
                    mask_model_gradients(self.model, self.grad_mask_lorsu)
                else:
                    mask_model_gradients_vision_tower(self.model, self.grad_mask_lorsu)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            
        metric_logger.synchronize_between_processes()
        self.lggr.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()}
    
    def log_stats(self, stats: Dict[str, float]):
        log_stats = {**{f"train_{k}": v for k, v in stats.items()}}
        self.lggr.info(log_stats)
        
    def prepare_model_eval(self):
        if 'lora' in self.training_args.ft_method: 
            assert isinstance(self.model, PeftModel), f"The model should be an instance of `PeftModel` but {type(self.model.__class__.__name__)} was given"
            self.model = self.model.merge_and_unload()
        elif 'lorsu' in self.training_args.ft_method:
            merge_weights(self.model, self.training_args)
                
        self.model.to(torch.float16)
        
    def train_eval(self):
        if self.data_args.is_cl:
            self.train_eval_cl()
        else:
            self.train_eval_offline()            
    
    def eval_vqa_datasets(self):
        self.prepare_model_eval()
        model_args, data_args, training_args = asdict(self.model_args), asdict(self.data_args), asdict(self.training_args)
        args = {**model_args, **data_args, **training_args}
        args = Namespace(**args)
        self.model.eval()

        gb = 1024.0 ** 3
        
        acc_tsivqa, num_samples_tsivqa, time_elapsed_tsivqa = eval_tsivqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"TSIVQA: acc={acc_tsivqa:.2f}%\t#Samples={num_samples_tsivqa}\tET={time_elapsed_tsivqa/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
            
        acc_gtsrb, num_samples_gtsrb, time_elapsed_gtsrb = eval_gtsrbvqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"GTSRB: acc={acc_gtsrb:.2f}%\t#Samples={num_samples_gtsrb}\tET={time_elapsed_gtsrb/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_aircraft, num_samples_aircraft, time_elapsed_aircraft = eval_aircraftvqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"FGVC-Aircraft: acc={acc_aircraft:.2f}%\t#Samples={num_samples_aircraft}\tET={time_elapsed_aircraft/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_eurosat, num_samples_eurosat, time_elapsed_eurosat = eval_eurosatvqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"EuroSAT: acc={acc_eurosat:.2f}%\t#Samples={num_samples_eurosat}\tET={time_elapsed_eurosat/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
                    
        acc_counteranimal, num_samples_counteranimal, time_elapsed_counteranimal = eval_counteranimalvqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"CounterAnimal: acc={acc_counteranimal:.2f}%\t#Samples={num_samples_counteranimal}\tET={time_elapsed_counteranimal/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_dalle, num_samples_dalle, time_elapsed_dalle = eval_dallevqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"DALLE-VQA: acc={acc_dalle:.2f}%\t#Samples={num_samples_dalle}\tET={time_elapsed_dalle/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_vsr, num_samples_vsr, time_elapsed_vsr = eval_vsr(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"VSR: acc={acc_vsr:.2f}%\t#Samples={num_samples_vsr}\tET={time_elapsed_vsr/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_hm, num_samples_hm, time_elapsed_hm = eval_hm(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"HM: acc={acc_hm:.2f}%\t#Samples={num_samples_hm}\tET={time_elapsed_hm/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
       
        acc_mmvp, num_samples_mmvp, time_elapsed_mmvp = eval_mmvpvqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"MMVP-VQA: acc={acc_mmvp:.2f}%\t#Samples={num_samples_mmvp}\tET={time_elapsed_mmvp/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")
        
        acc_visonlyqa, num_samples_visonlyqa, time_elapsed_visonlyqa = eval_visonlyqa(args, self.model, self.data_args.image_processor, self.tokenizer)
        gpu_memory = torch.cuda.max_memory_allocated() / gb
        self.lggr.info(f"VisOnlyQA: acc={acc_visonlyqa:.2f}%\t#Samples={num_samples_visonlyqa}\tET={time_elapsed_visonlyqa/60.:.1f} mins\tGPU memory allocated={gpu_memory:.1f} GB")