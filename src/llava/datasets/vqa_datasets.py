import os
import json
import pickle
from typing import Any, Dict, List, Iterator, Optional, Tuple, Callable, cast, Union
from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch

from torchvision.datasets import GTSRB as GTSRB_torch, FGVCAircraft as FGVCAircraft_torch
from torchvision.datasets.folder import default_loader, is_image_file

from PIL import Image

from datasets import load_dataset

import transformers

from llava.mm_utils import process_images
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


"""
>>> dataset_dev = ImageMCQDataset('MMBench_DEV_EN_V11')
>>> len(dataset_dev)
4876
>>> dataset_dev[0].keys()
dict_keys(['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'answer', 'category', 'image', 'l2-category', 'split'])

{'coarse_perception': 1381, 
'finegrained_perception (instance-level)': 1128, 
'finegrained_perception (cross-instance)': 667, 
'relation_reasoning': 637, 
'attribute_reasoning': 603, 
'logic_reasoning': 460}
"""


class VizWizEvalDatav2(torch.utils.data.Dataset):
    def __init__(self, root: Union[str, Path], vis_processor: transformers.CLIPImageProcessor, model_cfg):
        self.root = os.path.join(root, "vizwiz")
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.loaded_data = json.load(open(os.path.join(self.root, "Annotations", "val.json"), "r"))

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image']
        question = data['question']
        answers = data['answers']
        answers = '_'.join([answer['answer'] for answer in answers])
        image_path = os.path.join(self.root, "val", img_id)
        image = Image.open(image_path).convert('RGB')
        image = process_images([image], self.vis_processor, self.model_cfg)[0]
        question = f"The question is '{question}'. Based on the image, answer the question with a single word or phrase. and reply 'unanswerable' when the provided information is insufficient"
        
        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
            
        return image, question, answers
    
    
class HMEvalDatav2(torch.utils.data.Dataset):
    def __init__(self, root: Union[str, Path], vis_processor: transformers.CLIPImageProcessor, model_cfg):
        self.root = os.path.join(root, "hateful_memes")
        with open(os.path.join(self.root, "test_unseen.jsonl"), 'r') as jsonl_file:
            self.loaded_data = [json.loads(line) for line in jsonl_file]
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["img"]
        image_path = os.path.join(self.root, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = process_images([image], self.vis_processor, self.model_cfg)[0]
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? Answer yes or no. Answer:"
        labels = ann["label"]
        
        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question

        return image, question, labels
    

class ToyotaSmartHomeImagesVQA(torch.utils.data.Dataset):
    all_subjects = {"p02", "p03", "p04", "p06", "p07", "p09", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p25"}
    training_subjects = {"p03", "p04", "p06", "p07", "p09", "p12", "p13", "p15", "p17", "p19", "p25"}
    action_names = ['Cook.Cleandishes', 'Cook.Cleanup', 'Cook.Cut', 'Cook.Stir', 'Cutbread', 
                     'Drink.Frombottle', 'Drink.Fromcan', 'Drink.Fromcup', 'Drink.Fromglass', 
                     'Eat.Attable', 'Enter', 'Laydown', 'Leave', 'Makecoffee.Pourgrains', 
                     'Makecoffee.Pourwater', 'Maketea.Boilwater', 'Maketea.Insertteabag', 
                     'Pour.Frombottle', 'Pour.Fromcan', 'Pour.Fromkettle', 'Readbook', 
                     'Sitdown', 'Uselaptop', 'Usetablet', 'Usetelephone', 'Walk', 'WatchTV']
    def __init__(
        self,
        root: str,
        vis_processor: transformers.CLIPImageProcessor,
        model_cfg,
        train: bool = True, 
        session: int = -1,
    ) -> None:
        self.root = os.path.join(root, "toyota_smarthome_images")
        self.train = train
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.session = session
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        self.loaded_data = self.make_dataset()
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        if self.train:
            subjects_split = list(self.training_subjects)
        else:
            subjects_split = list(self.all_subjects - self.training_subjects)
            
        if self.session < 0:
            classes = self.action_names
        else:
            session_classes_list = [['WatchTV', 'Laydown', 'Sitdown', 'Pour.Fromkettle', 'Enter', 'Drink.Frombottle'],
                                        ['Eat.Attable', 'Pour.Frombottle', 'Cook.Cleandishes', 'Maketea.Boilwater', 'Leave', 'Cook.Cleanup'],
                                        ['Maketea.Insertteabag', 'Makecoffee.Pourwater', 'Drink.Fromcan', 'Readbook', 'Cutbread'],
                                        ['Drink.Fromcup', 'Drink.Fromglass', 'Usetablet', 'Pour.Fromcan', 'Usetelephone'],
                                        ['Walk', 'Cook.Stir', 'Makecoffee.Pourgrains', 'Cook.Cut', 'Uselaptop'],
                                    ]
            classes = session_classes_list[self.session]   
            
        with open(os.path.join(self.root, 'images_vqa_short_trunc.pkl'), 'rb') as fp:
           dict_vqa = pickle.load(fp)
                   
        instances = []
        all_image_names = [fname for fname in os.listdir(os.path.join(self.root, "images_trunc")) if fname.endswith('.jpg')]
        for fname in all_image_names:
            image_path = os.path.join(self.root, "images_trunc", fname)
            fname_split = fname.split("_")
            action = fname_split[0]
            subject = fname_split[1]
            if is_image_file(image_path) and subject in subjects_split and action in classes:
                dict_sample = {"img_path": image_path, "vqa_data": dict_vqa[fname]}
                instances.append(dict_sample)
                    
        return instances

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        image_path = sample["img_path"]
                
        question = data['question']
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]
        answer = data['answer']
        for option_num, option in enumerate(data['choices']):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
        return image, question, answer
    
    
class DalleVQA(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        vis_processor: transformers.CLIPImageProcessor,
        model_cfg,
    ) -> None:
        self.root = os.path.join(root, "dalle")
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.loaded_data = self.make_dataset()
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """            
        with open(os.path.join(self.root, 'dalle_vqa.pkl'), 'rb') as fp:
           dict_vqa = pickle.load(fp)
                   
        instances = []
        all_image_names = [fname for fname in os.listdir(os.path.join(self.root, "v2")) if fname.endswith('.png')]
        for fname in all_image_names:
                image_path = os.path.join(self.root, "v2", fname)
                if is_image_file(image_path):
                    dict_sample = {"img_path": image_path, "vqa_data": dict_vqa[fname]}
                    instances.append(dict_sample)
        return instances

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        image_path = sample["img_path"]
                
        question = data['question']
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]

        answer = data['answer']
        for option_num, option in enumerate(data['choices']):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer
    
    
class GTSRBVQA(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        vis_processor: transformers.CLIPImageProcessor,
        model_cfg,
        session: int = -1,
    ) -> None:
        self.root = root
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.session = session
        self.loaded_data = self.make_dataset()
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """    
        
        gtsrb_ds = GTSRB_torch(self.root, split='test')        
        with open(os.path.join(self.root, "gtsrb", 'gtsrb_vqa.pkl'), 'rb') as fp:
           dict_vqa = pickle.load(fp)
           
        classes = [25, 2, 11, 1, 40, 27, 5, 9, 17, 32, 29, 20, 39, 21, 15, 23, 10, 3, 18, 38,
                    42, 14, 22, 35, 34, 19, 33, 12, 26, 41, 0, 37, 6, 13, 24, 30, 28, 31, 7, 16, 4, 36, 8]
        if self.session >= 0:
            session_classes_list = np.array_split(classes, 5)
            classes = session_classes_list[self.session].tolist()        
            
        instances = []
        """
        all_image_names = [fname for fname in os.listdir(os.path.join(self.root, "gtsrb", "GTSRB", "Final_Test", "Images")) if fname.endswith('.ppm')]
        for fname in all_image_names:
            image_path = os.path.join(self.root, "gtsrb", "GTSRB", "Final_Test", "Images", fname)
            key_name = f"gtsrb@GTSRB@Final_Test@Images@{fname}"
            if is_image_file(image_path) and key_name in dict_vqa:
                dict_sample = {"img_path": image_path, "vqa_data": dict_vqa[key_name]}
                instances.append(dict_sample)
        """        
        for sample in gtsrb_ds._samples:        
            fname = sample[0].split(os.sep)[-1]
            key_name = f"gtsrb@GTSRB@Final_Test@Images@{fname}"
            if key_name in dict_vqa and sample[1] in classes:
                dict_sample = {"img_path": sample[0], "vqa_data": dict_vqa[key_name]}
                instances.append(dict_sample)
            
        return instances

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        image_path = sample["img_path"]
        
        question = 'What kind of traffic sign is this?'
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]

        answer = data['answer']
        for option_num, option in enumerate(data['choices']):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer, data['choices']
    
    
class FGVCAircraftVQA(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        vis_processor,
        model_cfg,
        session: int = -1,
    ) -> None:
        self.root = os.path.join(root, "fgvc_aircraft")
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.session = session
        self.loaded_data = self.make_dataset()
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """            
        with open(os.path.join(self.root, 'fgvc_aircraft_vqa.pkl'), 'rb') as fp:
           dict_vqa = pickle.load(fp)
        
        classes = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 
                    19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 
                    84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0,  72, 35, 
                    58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 
                    32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
        if self.session >= 0:
            session_classes_list = np.array_split(classes, 5)
            classes = session_classes_list[self.session].tolist()           
        instances = []
        ds_test = FGVCAircraft_torch(self.root, split='test')
        for i in range(len(ds_test)):
            if ds_test._labels[i] in classes:
                image_path = ds_test._image_files[i]
                key_name = image_path.split(os.sep)[-1]
                dict_sample = {"img_path": image_path, "vqa_data": dict_vqa[key_name]}
                instances.append(dict_sample)
        return instances   

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        image_path = sample["img_path"]
                
        question = 'What is the type of this aircraft?'
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]

        answer = data['answer']
        for option_num, option in enumerate(data['choices']):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer
    
    
class EuroSATVQA(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        vis_processor,
        model_cfg,
        session: int = -1,
    ) -> None:
        root = os.path.expanduser(root)    
        self.root = os.path.join(root, "eurosat")
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.session = session
        self.loaded_data = self.make_dataset()
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """            
        with open(os.path.join(self.root, 'eurosat_vqa.pkl'), 'rb') as fp:
           dict_vqa = pickle.load(fp)
           
        with open(os.path.join(self.root, 'eurosat_split.pkl'), 'rb') as handle:
            eurosat_data = pickle.load(handle)
        test_instances = eurosat_data['test']
        
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if self.session >= 0:
            session_classes_list = np.array_split(classes, 5)
            classes = session_classes_list[self.session].tolist()
                   
        instances = []
        for sample in test_instances:
            if sample[1] in classes:
                image_path = sample[0]
                key_name = image_path.split(os.sep)[-1]
                dict_sample = {"img_path": image_path, "vqa_data": dict_vqa[key_name]}
                instances.append(dict_sample)
        return instances   

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        image_path = sample["img_path"]
                
        question = 'What does this centered satellite photo show?'
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]

        answer = data['answer']
        for option_num, option in enumerate(data['choices']):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer
    
    
class MMVP_VQA(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        vis_processor,
        model_cfg,
    ) -> None:
        root = os.path.expanduser(root)    
        self.root = os.path.join(root, "MMVP")
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.loaded_data = self.make_dataset()
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> pd.DataFrame:
        """
        Generates a list of samples of a form (path_to_sample).
        """            
        instances_all = pd.read_csv(os.path.join(self.root, "Questions.csv"))
        indices = np.concatenate([np.arange(i, i+2) for i in range(2, len(instances_all), 4)])
        #indices = np.concatenate([np.arange(i, i+2) for i in range(0, len(instances_all), 4)]) for training
        instances = instances_all.iloc[indices]
        return instances   

    def __getitem__(self, index: int):
        sample = self.loaded_data.iloc[index]
        image_path = os.path.join(self.root, "MMVP-Images", f"{sample.Index}.jpg")
                
        question = sample.Question
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]
        
        answer = sample["Correct Answer"]
        question = question + '\n' + sample.Options + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer
    

class Vis_Only_QA(torch.utils.data.Dataset):
    def __init__(
        self,
        vis_processor,
        model_cfg,
    ) -> None:
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.loaded_data = self.make_dataset()
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> pd.DataFrame:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        real_eval = load_dataset("ryokamoi/VisOnlyQA_Eval_Real")
        synthetic_eval = load_dataset("ryokamoi/VisOnlyQA_Eval_Synthetic")
        
        instances = []
        for ds_eval in [real_eval, synthetic_eval]:
            for k in ds_eval.keys():
                ds_tmp = ds_eval[k]
                tmp_instance = [{'question': sample['question'],
                                 'answer': sample['answer'],
                                 'response_options': sample['response_options'],
                                 'decoded_image': sample['decoded_image']} for sample in ds_tmp if sample['question_type'] == 'single_answer']
                instances.extend(tmp_instance)
            
        return instances   

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
                
        question = sample['question']
        image = sample['decoded_image']
        image = process_images([image], self.vis_processor, self.model_cfg)[0]
        
        answer = sample["response_options"].index(sample["answer"])
        
        for option_num, option in enumerate(sample["response_options"]):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer
    
    
class CounterAnimalVQA(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        vis_processor: transformers.CLIPImageProcessor,
        model_cfg,
    ) -> None:
        self.root = os.path.join(root, "counteranimal", "LAION-final")
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg
        self.loaded_data = self.make_dataset()
        self.list_letters = ['A', 'B', 'C', 'D', 'E']
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """            
        with open(os.path.join(self.root, 'counter_animal_vqa_test.pkl'), 'rb') as fp:
           dict_test = pickle.load(fp)
           
        with open(os.path.join(self.root, 'counter_animal_ds.pkl'), 'rb') as fp:
           dict_ds = pickle.load(fp)
         
        ds_test = dict_ds["test_ds"]
        image_files_tmp, vqa_data_tmp = [], []
        for class_name in os.listdir(self.root):
            fold_dir = os.path.join(self.root, class_name)
            if os.path.isdir(fold_dir):
                for counter_name in os.listdir(fold_dir):
                    if "counter" in counter_name:
                        break
                assert "counter" in counter_name
                counter_dir = os.path.join(fold_dir, counter_name)
                images_name_list = [os.path.join(counter_dir, img_nm) for img_nm in ds_test[class_name]]
                
                vqa_class_list = [dict_test[f"{class_name}-{img_nm}"] for img_nm in ds_test[class_name]]
                image_files_tmp.extend(images_name_list)
                vqa_data_tmp.extend(vqa_class_list)
            
        perm = np.random.permutation(len(image_files_tmp))
        _image_files = [image_files_tmp[i] for i in perm]
        _vqa_data = [vqa_data_tmp[i] for i in perm]
                   
        instances = [{"img_path": _image_files[i], "vqa_data": _vqa_data[i]} for i in range(len(_image_files))]
        return instances
   
    def __getitem__(self, index: int):   
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        image_path = sample["img_path"]
                
        question = 'What is the species of this animal?'
        image = default_loader(image_path)
        image = process_images([image], self.vis_processor, self.model_cfg)[0]

        answer = data['answer']
        for option_num, option in enumerate(data['choices']):
            question = question + '\n' + self.list_letters[option_num] + '. ' + option
        question = question + '\n' + "Answer with the option's letter from the given choices directly."

        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
        return image, question, answer
    
    
class VSREvalDatav2(torch.utils.data.Dataset):
    def __init__(self, root: Union[str, Path], vis_processor: transformers.CLIPImageProcessor,  model_cfg):    
        self.images_path = os.path.join(root, "vsr", "images")
        self.loaded_data = load_dataset("cambridgeltl/vsr_zeroshot", split='test')
        self.vis_processor = vis_processor
        self.model_cfg = model_cfg

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_path = os.path.join(self.images_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = process_images([image], self.vis_processor, self.model_cfg)[0]
        question = ann["caption"]
        question = f'Based on the image, is this statement true or false? {question}'
        labels = 'true' if ann["label"] == 1 else 'false'
        
        if self.model_cfg.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question

        return image, question, labels
    
    
class ToyotaSmartHomeImagesVQAFT(torch.utils.data.Dataset):
    all_subjects = {"p02", "p03", "p04", "p06", "p07", "p09", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p25"}
    training_subjects = {"p03", "p04", "p06", "p07", "p09", "p12", "p13", "p15", "p17", "p19", "p25"}
    action_names = ['Cook.Cleandishes', 'Cook.Cleanup', 'Cook.Cut', 'Cook.Stir', 'Cutbread', 
                     'Drink.Frombottle', 'Drink.Fromcan', 'Drink.Fromcup', 'Drink.Fromglass', 
                     'Eat.Attable', 'Enter', 'Laydown', 'Leave', 'Makecoffee.Pourgrains', 
                     'Makecoffee.Pourwater', 'Maketea.Boilwater', 'Maketea.Insertteabag', 
                     'Pour.Frombottle', 'Pour.Fromcan', 'Pour.Fromkettle', 'Readbook', 
                     'Sitdown', 'Uselaptop', 'Usetablet', 'Usetelephone', 'Walk', 'WatchTV']
    
    def __init__(
        self,
        root: str,
        actions_list: List[str] = [],
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = os.path.join(root, "toyota_smarthome_images")
        self.few_shots = few_shots
        self.seed = seed
        assert all(act_name in self.action_names for act_name in actions_list), f"All the elements of `actions_list` should be in `action_names`!"
        self.actions_list = actions_list
        self.loaded_data = self.make_dataset()
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        subjects_split = list(self.training_subjects)
            
        with open(os.path.join(self.root, 'images_vqa_short_trunc.pkl'), 'rb') as fp:
           dict_vqa = pickle.load(fp)
           
        with open(os.path.join(self.root, 'images_text_description_short_trunc.pkl'), 'rb') as fp:
           dict_captions = pickle.load(fp)
           
        set_descr = set()
        for fname in dict_captions.keys():
            set_descr.add(dict_captions[fname])
                   
        instances = []
        all_image_names = [fname for fname in os.listdir(os.path.join(self.root, "images_trunc")) if fname.endswith('.jpg')]
        for fname in all_image_names:
                image_path = os.path.join(self.root, "images_trunc", fname)
                fname_split = fname.split("_")
                action = fname_split[0]
                subject = fname_split[1]
                #if is_image_file(image_path) and subject in subjects_split:
                if is_image_file(image_path) and subject in subjects_split and (action in self.actions_list or len(self.actions_list) == 0):    
                    dict_sample = {"img_path": image_path, "vqa_data": dict_vqa[fname], "caption": dict_captions[fname]}
                    instances.append(dict_sample)
                    
        if self.few_shots > 0:
            samples_per_action = {descr: [] for descr in set_descr} 
            for inst in instances:
                samples_per_action[inst["caption"]].append(inst["img_path"])
                
            few_shot_instances = []
            for seed_num, action in enumerate(samples_per_action.keys()):
                if min(len(samples_per_action[action]), self.few_shots) > 0:
                    list_samples_action = samples_per_action[action]
                    random.shuffle(list_samples_action)
                    sampled_list = list_samples_action[:min(len(samples_per_action[action]), self.few_shots)]
                    samples_rand = [{"img_path": image_path, "vqa_data": dict_vqa[image_path.split(os.sep)[-1]]} for image_path in sampled_list]
                    few_shot_instances.extend(samples_rand)
                    random.seed(self.seed + 2 * seed_num)
                    
            return few_shot_instances
        
        return instances
            
    def __getitem__(self, index) -> Tuple[str, str, str]:
        sample = self.loaded_data[index]
        data = sample["vqa_data"]
        answer = data['choices'][data['answer']]
        question = data["question"]

        return sample["img_path"], question, answer
        

class GTSRBVQAFT(torch.utils.data.Dataset):
    class_names = [
        'red and white circle 20 kph speed limit',
        'red and white circle 30 kph speed limit',
        'red and white circle 50 kph speed limit',
        'red and white circle 60 kph speed limit',
        'red and white circle 70 kph speed limit',
        'red and white circle 80 kph speed limit',
        'end / de-restriction of 80 kph speed limit',
        'red and white circle 100 kph speed limit',
        'red and white circle 120 kph speed limit',
        'red and white circle red car and black car no passing',
        'red and white circle red truck and black car no passing',
        'red and white triangle road intersection warning',
        'white and yellow diamond priority road',
        'red and white upside down triangle yield right-of-way',
        'stop',
        'empty red and white circle',
        'red and white circle no truck entry',
        'red circle with white horizonal stripe no entry',
        'red and white triangle with exclamation mark warning',
        'red and white triangle with black left curve approaching warning',
        'red and white triangle with black right curve approaching warning',
        'red and white triangle with black double curve approaching warning',
        'red and white triangle rough / bumpy road warning',
        'red and white triangle car skidding / slipping warning',
        'red and white triangle with merging / narrow lanes warning',
        'red and white triangle with person digging / construction / road work warning',
        'red and white triangle with traffic light approaching warning',
        'red and white triangle with person walking warning',
        'red and white triangle with child and person walking warning',
        'red and white triangle with bicyle warning',
        'red and white triangle with snowflake / ice warning',
        'red and white triangle with deer warning',
        'white circle with gray strike bar no speed limit',
        'blue circle with white right turn arrow mandatory',
        'blue circle with white left turn arrow mandatory',
        'blue circle with white forward arrow mandatory',
        'blue circle with white forward or right turn arrow mandatory',
        'blue circle with white forward or left turn arrow mandatory',
        'blue circle with white keep right arrow mandatory',
        'blue circle with white keep left arrow mandatory',
        'blue circle with white arrows indicating a traffic circle',
        'white circle with gray strike bar indicating no passing for cars has ended',
        'white circle with gray strike bar indicating no passing for trucks has ended',
    ]
    def __init__(
        self,
        root: str,
        classes_session: Optional[np.ndarray] = None,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.few_shots = few_shots
        self.seed = seed
        self.classes_session = classes_session
        self.samples = self.make_dataset()
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def _sample_n_samples_per_class(self, targets: np.ndarray):
        np.random.seed(self.seed)
        sampled_indices = []
        for class_label in np.unique(targets):
            class_indices = np.where(targets == class_label)[0]
            if len(class_indices) <= self.few_shots:
                sampled_indices.extend(class_indices)
            else:
                sampled_indices.extend(np.random.choice(
                    class_indices, self.few_shots, replace=False))
        return sampled_indices

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        ds = GTSRB_torch(self.root, split='train')
            
        if self.classes_session is None:
            image_files = [sample[0] for sample in ds._samples]
            targets = np.array([sample[1] for sample in ds._samples])
        else:
            image_files = [sample[0] for sample in ds._samples if sample[1] in self.classes_session]
            targets = np.array([sample[1] for sample in ds._samples if sample[1] in self.classes_session])
        
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            image_files_final = [image_files[i] for i in sampled_idx]
            targets_final = [targets[i] for i in sampled_idx]
        else:
            image_files_final, targets_final = image_files, targets
         
        instances = [(image_files_final[i], targets_final[i]) for i in range(len(targets_final))]
        
        return instances
    
    def __getitem__(self, index) -> Tuple[str, str, str]:
        image_path, target = self.samples[index]
        question = 'What kind of traffic sign is this?'
        answer = f"This is a {self.class_names[target]} traffic sign"
        
        return image_path, question, answer
    

class FGVCAircraftVQAFT(torch.utils.data.Dataset):    
    def __init__(
        self,
        root: str,
        classes_session: Optional[np.ndarray] = None,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = os.path.join(root, "fgvc_aircraft")
        self.few_shots = few_shots
        self.seed = seed
        self.classes_session = classes_session
        self.samples = self.make_dataset()
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def _sample_n_samples_per_class(self, targets: np.ndarray):
        np.random.seed(self.seed)
        sampled_indices = []
        for class_label in np.unique(targets):
            class_indices = np.where(targets == class_label)[0]
            if len(class_indices) <= self.few_shots:
                sampled_indices.extend(class_indices)
            else:
                sampled_indices.extend(np.random.choice(
                    class_indices, self.few_shots, replace=False))
        return sampled_indices

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        ds = FGVCAircraft_torch(self.root, split='trainval')
        if self.classes_session is None:
            image_files = ds._image_files
            targets = np.array(ds._labels)
        else:
            image_files = [img_file for i, img_file in enumerate(ds._image_files) if ds._labels[i] in self.classes_session]
            targets = np.array([label for label in ds._labels if label in self.classes_session])
        
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            image_files_final = [image_files[i] for i in sampled_idx]
            targets_final = [targets[i] for i in sampled_idx]
        else:
            image_files_final, targets_final = image_files, targets
         
        instances = [(image_files_final[i], targets_final[i]) for i in range(len(targets_final))]
        self.class_names = ds.classes

        return instances
            
    def __getitem__(self, index) -> Tuple[str, str, str]:
        image_path, target = self.samples[index]
        question = 'What is the type of this aircraft?'
        answer = f"The type of this aircraft is {self.class_names[target]}."

        return image_path, question, answer
    
    
class EuroSATVQAFT(torch.utils.data.Dataset):   
    class_names = [
        'annual crop land',
        'forest',
        'brushland or shrubland',
        'highway or road',
        'industrial buildings or commercial buildings',
        'pasture land',
        'permanent crop land',
        'residential buildings or homes or apartments',
        'river',
        'lake or sea',
    ]

    def __init__(
        self,
        root: str,
        classes_session: Optional[np.ndarray] = None,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        root = os.path.expanduser(root)    
        self.root = os.path.join(root, "eurosat")
        self.few_shots = few_shots
        self.seed = seed
        self.classes_session = classes_session
        self.samples = self.make_dataset()
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def _sample_n_samples_per_class(self, targets: np.ndarray):
        np.random.seed(self.seed)
        sampled_indices = []
        for class_label in np.unique(targets):
            class_indices = np.where(targets == class_label)[0]
            if len(class_indices) <= self.few_shots:
                sampled_indices.extend(class_indices)
            else:
                sampled_indices.extend(np.random.choice(
                    class_indices, self.few_shots, replace=False))
        return sampled_indices

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
           
        with open(os.path.join(self.root, 'eurosat_split.pkl'), 'rb') as handle:
            eurosat_data = pickle.load(handle)
        train_instances = eurosat_data['train']
        
        image_files_all = [sample[0] for sample in train_instances]
        targets_all = [sample[1] for sample in train_instances]
            
        if self.classes_session is None:
            image_files = image_files_all
            targets = np.array(targets_all)
        else:
            image_files = [img_file for i, img_file in enumerate(image_files_all) if targets_all[i] in self.classes_session]
            targets = np.array([label for label in targets_all if label in self.classes_session])
        
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            image_files_final = [image_files[i] for i in sampled_idx]
            targets_final = [targets[i] for i in sampled_idx]
        else:
            image_files_final, targets_final = image_files, targets
         
        instances = [(image_files_final[i], targets_final[i]) for i in range(len(targets_final))]

        return instances
            
    def __getitem__(self, index) -> Tuple[str, str, str]:
        image_path, target = self.samples[index]
        question = 'What does this centered satellite photo show?'
        answer = f"It shows a {self.class_names[target]}."

        return image_path, question, answer
    
    
class CounterAnimalVQAFT(torch.utils.data.Dataset):    
    def __init__(
        self,
        root: str,
        classes_session: Optional[np.ndarray] = None,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        assert few_shots >= 0, f"few_shots argument should be a non-negative integer but {few_shots} was given"
        assert seed >= 0, f"few_shots argument should be a non-negative integer but {few_shots} was given"
        self.root = os.path.join(root, "counteranimal", "LAION-final")
        self.few_shots = few_shots
        self.seed = seed
        self.classes_session = classes_session
        self.samples = self.make_dataset()
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def _sample_n_samples_per_class(self, targets: np.ndarray):
        np.random.seed(self.seed)
        sampled_indices = []
        for class_label in np.unique(targets):
            class_indices = np.where(targets == class_label)[0]
            if len(class_indices) <= self.few_shots:
                sampled_indices.extend(class_indices)
            else:
                sampled_indices.extend(np.random.choice(
                    class_indices, self.few_shots, replace=False))
        return sampled_indices

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        with open(os.path.join(self.root, 'counter_animal_ds.pkl'), 'rb') as fp:
           dict_all = pickle.load(fp)
           
        ds = dict_all["train_ds"]
        image_files_tmp, targets_tmp = [], []
        for class_name in os.listdir(self.root):
            fold_dir = os.path.join(self.root, class_name)
            if os.path.isdir(fold_dir):
                for counter_name in os.listdir(fold_dir):
                    if "counter" in counter_name:
                        break
                assert "counter" in counter_name
                counter_dir = os.path.join(fold_dir, counter_name)
                images_name_list = [os.path.join(counter_dir, img_nm) for img_nm in ds[class_name]]
                
                true_label_int = int(class_name.split(" ")[0])
                label_class_list = len(images_name_list) * [true_label_int]
                
                image_files_tmp.extend(images_name_list)
                targets_tmp.extend(label_class_list)
        
        np.random.seed(self.seed)       
        perm = np.random.permutation(len(image_files_tmp))
        _image_files = [image_files_tmp[i] for i in perm]
        _labels = [targets_tmp[i] for i in perm]
        
        if self.classes_session is None:
            image_files = _image_files
            targets = np.array(_labels)
        else:
            image_files = [img_file for i, img_file in enumerate(_image_files) if _labels[i] in self.classes_session]
            targets = np.array([label for label in _labels if label in self.classes_session])
        
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            image_files_final = [image_files[i] for i in sampled_idx]
            targets_final = [targets[i] for i in sampled_idx]
        else:
            image_files_final, targets_final = image_files, targets
         
        instances = [(image_files_final[i], targets_final[i]) for i in range(len(targets_final))]
        self.class_names = []
        for line in open(os.path.join(self.root, "imagenet_names.txt")):
            name = line.split('\t')[-1].rstrip()
            self.class_names.append(name)

        return instances
    
    def __getitem__(self, index) -> Tuple[str, str, str]:
        image_path, target = self.samples[index]
        answer = f"This is a {self.class_names[target]}."
        question = 'What is the species of this animal?'

        return image_path, question, answer
    
    
class VSRFT(torch.utils.data.Dataset):        
    def __init__(
        self,
        root: Union[str, Path],
        objects_list: Optional[List[str]] = None,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        self.images_path = os.path.join(root, "vsr", "images")
        self.few_shots = few_shots
        self.seed = seed
        self.objects_list = objects_list
        self.samples = self.make_dataset()
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        loaded_data = load_dataset("cambridgeltl/vsr_zeroshot", split='train')
        list_tuples = [(os.path.join(self.images_path, ann["image"]), ann["caption"], ann["label"]) 
                       for ann in loaded_data] # image, caption, labels
        
        if self.objects_list is not None:
            instances = [tpl for i, tpl in enumerate(list_tuples) if loaded_data[i]['obj'] in self.objects_list]
        else: 
            instances = list_tuples
           
        if self.few_shots > 0:
            random.seed(self.seed)
            random.shuffle(instances)
            if self.few_shots < 6:
                instances = instances[:100]
            elif self.few_shots < 21:
                instances = instances[:300]
            else:
                instances = instances[:600]

        return instances
            
    def __getitem__(self, index) -> Tuple[str, str, str]:
        image_path, caption, label = self.samples[index]
        answer = 'true' if label == 1 else 'false'
        question = f'Based on the image, is this statement true or false? {caption}'

        return image_path, question, answer
    
    
class HMTrainFT(torch.utils.data.Dataset):   
    def __init__(
        self,
        root: str,
        session: int = -1,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = os.path.join(root, "hateful_memes")
        self.few_shots = few_shots
        self.seed = seed
        self.session = session
        self.samples = self.make_dataset()
        
    def __len__(self) -> int:
        return len(self.samples)

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
           
        with open(os.path.join(self.root, "train.jsonl"), 'r') as jsonl_file:
            loaded_data_all = [json.loads(line) for line in jsonl_file]
        
        if self.session >= 0:
            assert self.session <= 4
            with open(os.path.join(self.root, 'data_per_session.pkl'), 'rb') as handle:
                session_data = pickle.load(handle)[self.session]
                
            loaded_data = [loaded_data_all[i] for i in session_data]
        else:
            loaded_data = loaded_data_all
            
        if self.few_shots > 0:
            if self.few_shots < 6:
                loaded_data = loaded_data[:100]
            elif self.few_shots < 21:
                loaded_data = loaded_data[:300]
            else:
                loaded_data = loaded_data[:600]

        return loaded_data
            
    def __getitem__(self, index) -> Tuple[str, str, str]:
        ann = self.samples[index]
        image_id = ann["img"]
        image_path = os.path.join(self.root, f"{image_id}")
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? (Answer yes or no)"
        answer = "yes" if ann["label"] == 1 else "no"
        
        return image_path, question, answer
    
    
class MMVP_VQA_FT(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        session: int = -1,
    ) -> None:
        root = os.path.expanduser(root)    
        self.root = os.path.join(root, "MMVP")
        self.session = session
        self.loaded_data = self.make_dataset()        
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> pd.DataFrame:
        """
        Generates a list of samples of a form (path_to_sample).
        """            
        instances_all = pd.read_csv(os.path.join(self.root, "Questions.csv"))
        indices = np.concatenate([np.arange(i, i+2) for i in range(0, len(instances_all), 4)])
        instances = instances_all.iloc[indices]
        
        if self.session >= 0:
            idx_l = self.session * 30
            idx_u = idx_l + 30
            instances = instances.iloc[idx_l:idx_u]
            
        return instances   

    def __getitem__(self, index: int):
        sample = self.loaded_data.iloc[index]
        image_path = os.path.join(self.root, "MMVP-Images", f"{sample.Index}.jpg")
                
        question = f"{sample.Question}\n{sample.Options}\nAnswer with the option's letter from the given choices directly."
        answer = sample["Correct Answer"]
                
        return image_path, question, answer
    
    
class Vis_Only_QAFT(torch.utils.data.Dataset):
    def __init__(
        self,
        session: int = -1,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        assert -1 <= session < 5, f"Arguments session should be an integer in the interval -1 <= x < 5 but {session} was given!"
        self.session = session
        self.few_shots = few_shots
        self.seed = seed
        self.loaded_data = self.make_dataset()
        
    def __len__(self):
        return len(self.loaded_data)

    def make_dataset(self) -> pd.DataFrame:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        if self.session < 0:
            """
            for k in train_ds.keys():
                ds_tmp = train_ds[k]
                rand_perm = np.random.permutation(len(ds_tmp))
                for i in range(len(ds_tmp)):
                     sample_perm = ds_tmp[int(rand_perm[i])]
                     if sample_perm['question_type'] == 'single_answer':
                        tmp_sample = {'prompt_no_reasoning': sample_perm['prompt_no_reasoning'],
                                    'answer': sample_perm['answer'],
                                    'response_options': sample_perm['response_options'],
                                    'decoded_image': sample_perm['decoded_image']
                                    }
                        instances.append(tmp_sample)
            """
            with open(os.path.join(self.root, 'all_instances.pkl'), 'rb') as fp:
                instances = pickle.load(fp)
        else:
            np.random.seed(self.seed)
            train_ds = load_dataset("ryokamoi/VisOnlyQA_Train")
            instances = []
            category_list = ['syntheticgeometry__triangle', 'syntheticgeometry__quadrilateral', 'syntheticgeometry__length', 'syntheticgeometry__angle', 'syntheticgeometry__area', '3d__size', '3d__angle']
            list_cat_session = ['syntheticgeometry__area', '3d__size', '3d__angle'] if self.session == 4 else [category_list[self.session]]
            num_points = self.few_shots if self.few_shots > 0 else 10000
            for k in list_cat_session:
                ds_tmp = train_ds[k]
                rand_perm = np.random.permutation(len(ds_tmp))
                num_points_final = min(num_points, len(ds_tmp))
                for i in range(num_points_final):
                     sample_perm = ds_tmp[int(rand_perm[i])]
                     if sample_perm['question_type'] == 'single_answer':
                        tmp_sample = {'prompt_no_reasoning': sample_perm['prompt_no_reasoning'],
                                    'answer': sample_perm['answer'],
                                    'response_options': sample_perm['response_options'],
                                    'decoded_image': sample_perm['decoded_image']
                                    }
                        instances.append(tmp_sample)            
        return instances   

    def __getitem__(self, index: int):
        sample = self.loaded_data[index]
                
        question = sample['prompt_no_reasoning']
        image = sample['image_path'] if self.session < 0 else sample['decoded_image']     
        answer = sample["answer"]
                
        return image, question, answer