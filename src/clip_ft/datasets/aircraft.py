import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np

from torch import Tensor

from torchvision.datasets.folder import  default_loader, is_image_file
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import FGVCAircraft as FGVCAircraft_torch

from open_clip.tokenizer import HFTokenizer


class Aircraft(VisionDataset):
    templates = ['a photo of a {}, a type of aircraft.',
                 'a photo of the {}, a type of aircraft.',
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
        ds = FGVCAircraft_torch(self.root, split='trainval') if self.train else FGVCAircraft_torch(self.root, split='test')
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
    
    
class Aircraft_labeled(Aircraft):
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
    