import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np

from torch import Tensor

from torchvision.datasets.folder import  default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import FGVCAircraft as FGVCAircraft_torch

from open_clip.tokenizer import HFTokenizer


class FGVCAircraft(VisionDataset):
    class_names = [
        '707-320',             
        '727-200',
        '737-200',
        '737-300',
        '737-400',
        '737-500',
        '737-600',
        '737-700',
        '737-800',
        '737-900',
        '747-100',
        '747-200',
        '747-300',
        '747-400',
        '757-200',
        '757-300',
        '767-200',
        '767-300',
        '767-400',
        '777-200',
        '777-300',
        'A300B4',
        'A310',
        'A318',
        'A319',
        'A320',
        'A321',
        'A330-200',
        'A330-300',
        'A340-200',
        'A340-300',
        'A340-500',
        'A340-600',
        'A380',
        'ATR-42',
        'ATR-72',
        'An-12',
        'BAE 146-200',
        'BAE 146-300',
        'BAE-125',
        'Beechcraft 1900',
        'Boeing 717',
        'C-130',
        'C-47',
        'CRJ-200',
        'CRJ-700',
        'CRJ-900',
        'Cessna 172',
        'Cessna 208',
        'Cessna 525',
        'Cessna 560',
        'Challenger 600',
        'DC-10',
        'DC-3',
        'DC-6',
        'DC-8',
        'DC-9-30',
        'DH-82',
        'DHC-1',
        'DHC-6',
        'DHC-8-100',
        'DHC-8-300',
        'DR-400',
        'Dornier 328',
        'E-170',
        'E-190',
        'E-195',
        'EMB-120',
        'ERJ 135',
        'ERJ 145',
        'Embraer Legacy 600',
        'Eurofighter Typhoon',
        'F-16A/B',
        'F/A-18',
        'Falcon 2000',
        'Falcon 900',
        'Fokker 100',
        'Fokker 50',
        'Fokker 70',
        'Global Express',
        'Gulfstream IV',
        'Gulfstream V',
        'Hawk T1',
        'Il-76',
        'L-1011',
        'MD-11',
        'MD-80',
        'MD-87',
        'MD-90',
        'Metroliner',
        'Model B200',
        'PA-28',
        'SR-20',
        'Saab 2000',
        'Saab 340',
        'Spitfire',
        'Tornado',
        'Tu-134',
        'Tu-154',
        'Yak-42',
    ]

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
            targets = np.array([label for label in ds._labels])
        else:
            targets = np.array([label for label in ds._labels if label in self.classes_session])
            
        if self.few_shots > 0:
            sampled_idx = self._sample_n_samples_per_class(targets)
            image_files = [ds._image_files[i] for i in sampled_idx]
        elif self.classes_session is not None:
            image_files = [img_file for i, img_file in enumerate(ds._image_files) if targets[i] in self.classes_session]
        else:
            image_files = ds._image_files
            
        instances = [(image_files[i], targets[i]) for i in range(len(image_files))]

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
            
        caption = random.choice(self.templates).format(self.class_names[target])
        text = self.tokenizer(caption)[0]
            
        return img, text

    def __len__(self) -> int:
        return len(self.samples)
    
    
class FGVCAircraft_labeled(FGVCAircraft):
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
    