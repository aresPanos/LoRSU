import os
import argparse
from argparse import Namespace
import random
from typing import Tuple

import datetime

import numpy as np

import torch


def parsed_args_eval() -> Namespace:
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--dataroot', type=str, default="/home/ap2313/rds/hpc-work/datasets")
    parser.add_argument('--conv_mode', type=str, default='vicuna_v1', choices=["v0", "v1", "vicuna_v1", "llama_2", "mistral_instruct",
                                                                                "chatml_direct", "mistral_direct", "plain", "v0_plain", "llava_v0", 
                                                                                "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"])
    parser.add_argument("--model_path", default="/home/ap2313/rds/hpc-work/code/LLaVA/llava_models/llava-v1.5-7b-lora", help="Path to fine-tuned Vicuna-v1.5-7b model LoRA parameters.")    
    parser.add_argument("--model_base", type=str, default="/home/ap2313/rds/hpc-work/code/LLaVA/llava_models/vicuna-7b-v1-5",  help="Path to the pretrained Vicuna-v1.5-7b model.")
    parser.add_argument("--model_name", type=str, default="llava-vicuna-7b-v1-5",  help="Model name.")
    parser.add_argument("--sep", type=str,  default=",",  help="Sep for conversation")
    parser.add_argument("--num_beams", type=int, default=1)
    #parser.add_argument('--zero_shot', action='store_true', default=False, help='Whether to use the zero-shot model or not.')
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--fine_tuned_clip_model", type=str, default=None, help="Path to the fine-tuned CLIP-ViT-L-14-336 model.")
    parser.add_argument("--checkpoint_format_clip_model", type=str, default=None, help="Path to the fine-tuned CLIP-ViT-L-14-336 model.")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="max number of generated tokens")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--log_folder", type=str, default="/home/ap2313/rds/hpc-work/code/LLaVA/logs", help="Directory of the log files.")
    parser.add_argument('--dataloader_num_workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2024)
    parsed_args = parser.parse_args()
    parsed_args.device = get_device()

    return parsed_args
            
            
def set_seed(seed: int):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        
def get_device() -> torch.device:
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(conv.roles[0], text) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]

    return texts


def extract_details_from_path(ft_path: str) -> Tuple[str, str]:
    assert os.path.isfile(ft_path), f"There is no file `{ft_path}`"
    splitted_dir = ft_path.split(os.sep)
    if "spu_attn" in ft_path:
        train_mode, train_details = splitted_dir[-6], f"{os.sep}".join(splitted_dir[-5:-1])
    elif any(method in ft_path for method in ["lora", "spu", "last_mlps", "adalora"]):
        train_mode, train_details = splitted_dir[-4], f"{os.sep}".join(splitted_dir[-3:-1])
    else:
        train_mode, train_details = splitted_dir[-3], splitted_dir[-2]
    
    return train_mode, train_details


def create_logname(args: Namespace, zero_shot_forgetting: bool = False) -> str:
    assert os.path.isdir(args.log_folder), f"`{args.log_folder}` is not valid directory."    
    date_format = "%Y%m%d-%H%M%S"
    now_timestamp = datetime.datetime.now().strftime(date_format)
     
    is_forgetting = args.checkpoint_format_clip_model is not None
    if args.fine_tuned_clip_model or is_forgetting:
        ft_clip_path = args.checkpoint_format_clip_model if args.checkpoint_format_clip_model else args.fine_tuned_clip_model
            
        if "tsi" in ft_clip_path:
            dataset = "tsi"  
        elif "gtsrb" in ft_clip_path:
            dataset = "gtsrb" 
        elif "aircraft" in ft_clip_path:
            dataset = "aircraft" 
        elif "eurosat" in ft_clip_path:
            dataset = "eurosat" 
        elif "counteranimal" in ft_clip_path:
            dataset = "counteranimal" 
        elif "vsr" in ft_clip_path:
            dataset = "vsr"
        elif "hm" in ft_clip_path:
            dataset = "hm" 
        elif "visonlyqa" in ft_clip_path:
            dataset = "visonlyqa"
        elif "CL-5d" in ft_clip_path:
            dataset = ""
        else:
            raise ValueError(f"Invalid fine-tuning dataset in path {ft_clip_path}")
        
        if is_forgetting :
            epochs = int(args.checkpoint_format_clip_model.split("epochs-")[1].split("_")[0])
            ft_clip_path = args.checkpoint_format_clip_model.format(epochs, 5)
            
        train_mode, train_folder = extract_details_from_path(ft_clip_path)
        setting = "offline" if "offline" in ft_clip_path else "CL"
        setting = "CL-5d" if "CL-5d" in ft_clip_path else setting
        fs_num = ""
        if setting == "CL":
            split_path = ft_clip_path.split(os.sep)
            found = False
            for fs_num in split_path:
                if "FS-" in fs_num:
                    found = True
                    break
            assert found, f"There is no FS-`num_shots` in path {ft_clip_path}"     
                
        train_mode = os.path.join(setting, dataset, fs_num, train_mode)
    else:
        train_mode, train_folder = "zero_shot", ""
        if zero_shot_forgetting:
            train_mode += "_forgetting"
            
        
    log_dir = os.path.join(args.log_folder, "llava_vicuna_7b", train_mode, train_folder)
    os.makedirs(log_dir, exist_ok=True)
    if is_forgetting or zero_shot_forgetting:
        log_fname = os.path.join(log_dir, f"forgetting_log_{now_timestamp}.txt")
    else:
        log_fname = os.path.join(log_dir, f"all_vqa_eval_log_{now_timestamp}.txt")
    
    return log_fname