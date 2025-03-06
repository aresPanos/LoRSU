import torch
from PIL import Image
import os
import json
from pathlib import Path
import random
import pickle
from typing import Any, Callable, Optional, Tuple, Union, List, Dict

import pandas as pd
import numpy as np

from open_clip.tokenizer import HFTokenizer

from datasets import load_dataset
   
class VSRTrain(torch.utils.data.Dataset):        
    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable,
        tokenizer: HFTokenizer,
        objects_list: Optional[List[str]] = None,
        few_shots: Optional[int] = 0,
        seed: int = 0,
    ) -> None:
        self.images_path = os.path.join(root, "vsr", "images")
        self.objects_list = objects_list
        self.few_shots = few_shots
        self.seed = seed
        self.transform = transform
        self.tokenizer = tokenizer
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
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        text = f"Based on the image, the statement '{caption}' is {answer}"
        text = self.tokenizer(text)[0]

        return image, text
    
    
class HMTrain(torch.utils.data.Dataset):   
    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable,
        tokenizer: HFTokenizer,
        session: int = -1,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = os.path.join(root, "hateful_memes")
        self.transform = transform
        self.tokenizer = tokenizer
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
        
        if self.session >=0:
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
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        text = f"This is a hateful image writting '{question}'" if ann["label"] == 1 else f"This is an image writting '{question}'"
        text = self.tokenizer(text)[0]

        return image, text
    

class VisOnlyTrain(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable,
        tokenizer: HFTokenizer,
        session: int = -1,
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        assert -1 <= session < 5, f"Arguments session should be an integer in the interval -1 <= x < 5 but {session} was given!"
        self.root = os.path.join(root, "VisOnlyQA", "VisOnlyQA_Train")
        self.transform = transform
        self.tokenizer = tokenizer
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
        answer = sample["answer"]        
        if self.session < 0:
            image_path = sample['image_path'] 
            image = Image.open(image_path).convert("RGB")
        else:
            image = sample['decoded_image']     

        
        image = self.transform(image)
        
        text = f"{question}\nThe answer is {answer}"
        text = self.tokenizer(text)[0]
                
        return image, text
        