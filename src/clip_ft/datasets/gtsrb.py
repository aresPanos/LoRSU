import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np

from torch import Tensor

from torchvision.datasets.folder import  default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import GTSRB as GTSRB_torch

from open_clip.tokenizer import HFTokenizer


class GTSRB(VisionDataset):
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

    templates = [
        'a zoomed in photo of a "{}" traffic sign.',
        'a centered photo of a "{}" traffic sign.',
        'a close up photo of a "{}" traffic sign.',
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
        ds = GTSRB_torch(self.root, split='train') if self.train else GTSRB_torch(self.root, split='test')
        """
        if self.classes_session is None:
            targets = np.array([i[1] for i in ds._samples])
        else:
            targets = np.array([i[1] for i in ds._samples if i[1] in self.classes_session])
            
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            instances = [ds._samples[i] for i in sampled_idx]
        elif self.classes_session is not None:
            instances = [sample for sample in ds._samples if sample[1] in self.classes_session]
        else:
            instances = ds._samples
        """    
            
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
    
    
class GTSRB_labeled(GTSRB):
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
    