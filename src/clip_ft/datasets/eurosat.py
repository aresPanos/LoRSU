import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np

from torch import Tensor

from torchvision.datasets.folder import  default_loader
from torchvision.datasets.vision import VisionDataset

from open_clip.tokenizer import HFTokenizer


class EuroSAT(VisionDataset):
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

    templates = [
        'a centered satellite photo of {}.',
        'a centered satellite photo of a {}.',
        'a centered satellite photo of the {}.',
    ]
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            tokenizer: Optional[HFTokenizer] = None,
            classes_session: Optional[np.ndarray] = None,
            few_shots: int = 0,
            seed: int = 0,
        ) -> None:
            super().__init__(root, transform=transform)
            assert few_shots >= 0, f"few_shots argument should be a non-negative integer but {few_shots} was given"
            assert seed >= 0, f"few_shots argument should be a non-negative integer but {few_shots} was given"
            self.root = root
            self.transform = transform
            self.train = train
            self.tokenizer = tokenizer
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
        with open(os.path.join(self.root, "eurosat", 'eurosat_split.pkl'), 'rb') as handle:
            eurosat_data = pickle.load(handle)
        
        ds_instances = eurosat_data['train'] if self.train else eurosat_data['test']
            
        if self.classes_session is None:
            image_files = [sample[0] for sample in ds_instances]
            targets = np.array([sample[1] for sample in ds_instances])
        else:
            image_files = [sample[0] for sample in ds_instances if sample[1] in self.classes_session]
            targets = np.array([sample[1] for sample in ds_instances if sample[1] in self.classes_session])
        
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            image_files_final = [image_files[i] for i in sampled_idx]
            targets_final = [targets[i] for i in sampled_idx]
        else:
            image_files_final, targets_final = image_files, targets
         
        instances = [(image_files_final[i], targets_final[i]) for i in range(len(targets_final))]
        
        return instances
        

    def __getitem__(self, index: int) -> Tuple[Tensor, Union[Tensor, List[Tensor]]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
            
        caption =  random.choice(self.templates).format(self.class_names[target])
        text = self.tokenizer(caption)[0]
            
        return img, text

    def __len__(self) -> int:
        return len(self.samples)
    
    
class EuroSAT_labeled(EuroSAT):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        few_shots: int = 0,
        classes_session: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, few_shots=few_shots, classes_session=classes_session)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

        