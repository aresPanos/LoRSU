import os
from argparse import Namespace
from typing import Tuple, Dict, Any, Union
import argparse

import numpy as np

import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import transformers

from llava.datasets.vqa_datasets import VizWizEvalDatav2, VSREvalDatav2, HMEvalDatav2, ToyotaSmartHomeImagesVQA, \
                                        DalleVQA, GTSRBVQA, FGVCAircraftVQA, CounterAnimalVQA, EuroSATVQA, MMVP_VQA, Vis_Only_QA

from llava.eval.eval_utils import prepare_texts
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token


def eval_vizwiz(args: Namespace, model: LlavaLlamaForCausalLM, vis_processor: transformers.CLIPImageProcessor, tknzr: transformers.PreTrainedTokenizer) -> Tuple[float, int, float]:
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = VizWizEvalDatav2(args.dataroot, vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    total_acc = []
    for images, texts, gt_answers in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )
                answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            count = 0
            gt_answer = gt_answers[i]
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
    toc = time.time() - tic       
    
    return 100. * np.average(total_acc), len(total_acc), toc  


def eval_tsivqa(args: Namespace, 
                model: LlavaLlamaForCausalLM, 
                vis_processor: transformers.CLIPImageProcessor, 
                tknzr: transformers.PreTrainedTokenizer,
                session: int = -1) -> Tuple[float, int, float]:
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = ToyotaSmartHomeImagesVQA(args.dataroot, vis_processor, model.config, False, session)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1         
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc  


def eval_dallevqa(args: Namespace, 
                  model: LlavaLlamaForCausalLM, 
                  vis_processor: transformers.CLIPImageProcessor, 
                  tknzr: transformers.PreTrainedTokenizer) -> Tuple[float, int, float]:
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = DalleVQA(args.dataroot, vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1    
                                
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc  


def eval_gtsrbvqa(args: Namespace, 
                  model: LlavaLlamaForCausalLM, 
                  vis_processor: transformers.CLIPImageProcessor, 
                  tknzr: transformers.PreTrainedTokenizer, 
                  session: int = -1) -> Tuple[float, int, float]:
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = GTSRBVQA(args.dataroot, vis_processor, model.config, session)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels, choices in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer in ["a", "b", "c", "d", "e", 
                          "a.", "b.", "c.", "d.", "e.",
                          "a.", "B.", "C.", "D.", "E."]:
                answer = answer.upper()[0]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1          
            else:
                if any(f"{k}." in answer for k in dict_letters.keys()):
                    for k in dict_letters.keys():
                        if f"{k}." in answer:
                            if dict_letters[k] == label:
                                count += 1    
                            break
                else:
                    print(f"\n\ntext: {text}")  
                    print(f"label: {label}") 
                    #print("choices_i:", choices[i]) 
                    print(f"LLaVA answer: {answer}")   
                      
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc  


def eval_aircraftvqa(args: Namespace, 
                     model: LlavaLlamaForCausalLM, 
                     vis_processor: transformers.CLIPImageProcessor, 
                     tknzr: transformers.PreTrainedTokenizer,
                     session: int = -1) -> Tuple[float, int, float]:
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = FGVCAircraftVQA(args.dataroot, vis_processor, model.config, session)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1              
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc 

def eval_eurosatvqa(args: Namespace,
                    model: LlavaLlamaForCausalLM, 
                    vis_processor: transformers.CLIPImageProcessor, 
                    tknzr: transformers.PreTrainedTokenizer,
                    session: int = -1) -> Tuple[float, int, float]:
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = EuroSATVQA(args.dataroot, vis_processor, model.config, session)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1              
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc 


def eval_mmvpvqa(args: Namespace, model: LlavaLlamaForCausalLM, vis_processor: transformers.CLIPImageProcessor, tknzr: transformers.PreTrainedTokenizer) -> Tuple[float, int, float]:
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = MMVP_VQA(args.dataroot, vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            
            if 'Yes' in answer or 'yes' in answer:
                answer = "(a)"
            elif 'No' in answer or 'no' in answer:
                answer = "(b)"
            elif '(a)' in answer:
                answer = "(a)"
            elif '(b)' in answer:
                answer = "(b)"
            elif len(answer) < 3:
                answer = f"({answer.lower()})"
                
            if label in answer:
                count += 1
            elif answer not in ['(a)', '(b)']:
                print(f"\nQuestion:\n{text}\nLLM response: {answer}\n True answer: {label}") 
                           
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc 


def eval_visonlyqa(args: Namespace, model: LlavaLlamaForCausalLM, vis_processor: transformers.CLIPImageProcessor, tknzr: transformers.PreTrainedTokenizer) -> Tuple[float, int, float]:
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = Vis_Only_QA(vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip().upper()
            label = labels[i]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1
            else:
                print(f"\nQuestion:\n{text}\nLLM response: {answer}\nTrue answer: {label}")             
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc 


def eval_counteranimalvqa(args: Namespace, 
                          model: LlavaLlamaForCausalLM, 
                          vis_processor: transformers.CLIPImageProcessor, 
                          tknzr: transformers.PreTrainedTokenizer,
                          session: int = -1) -> Tuple[float, int, float]:
    dict_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = CounterAnimalVQA(args.dataroot, vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    tic = time.time()    
    count = 0
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer in dict_letters:
                if dict_letters[answer] == label:
                    count += 1          
    toc = time.time() - tic    
       
    return 100. * count / len(ds), len(ds), toc 


def eval_vsr(args: Namespace, model: LlavaLlamaForCausalLM, vis_processor: transformers.CLIPImageProcessor, tknzr: transformers.PreTrainedTokenizer) -> Tuple[float, int, float]:
    conv_temp = conv_templates[args.conv_mode].copy()
    ds = VSREvalDatav2(args.dataroot, vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    count = 0
    total = 0
    tic = time.time()    
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(input_ids,
                                            images=image,
                                            image_sizes=[image[0].size],
                                            do_sample=True if args.temperature > 0 else False,
                                            temperature=args.temperature,
                                            top_p=args.top_p,
                                            num_beams=args.num_beams,
                                            # no_repeat_ngram_size=3,
                                            #max_new_tokens=1024,
                                            max_new_tokens=args.max_new_tokens,
                                            use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()
            label = labels[i]
            if label in answer:
                count += 1
            total += 1
    toc = time.time() - tic  
    
    return 100. * count / total, total, toc  


def eval_hm(args: Namespace, model: LlavaLlamaForCausalLM, vis_processor: transformers.CLIPImageProcessor, tknzr: transformers.PreTrainedTokenizer) -> Tuple[float, int, float]:
    conv_temp = conv_templates[args.conv_mode].copy()
    
    ds = HMEvalDatav2(args.dataroot, vis_processor, model.config)
    eval_dataloader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
    
    count = 0
    total = 0
    tic = time.time()    
    for images, texts, labels in tqdm(eval_dataloader):
        images = images.to(device=args.device, dtype=torch.float16, non_blocking=True)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        for i, text in enumerate(texts):
            image = images[i].unsqueeze(0)
            with torch.inference_mode():
                input_ids = tokenizer_image_token(text, tknzr, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    image_sizes=[image[0].size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    #max_new_tokens=1024,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            answer = tknzr.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            label = labels[i]
            if answer.lower().strip().replace(".", "") == "yes":
                answer = 1
            elif answer.lower().strip().replace(".", "") == "no":
                answer = 0
            else:
                print("non-matching answer", answer)

            if answer == label:
                count += 1
            total += 1
            
    toc = time.time() - tic  
         
    return 100. * count / total, total, toc  