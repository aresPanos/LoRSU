import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np

from torch import Tensor

from torchvision.datasets.folder import  default_loader
from torchvision.datasets.vision import VisionDataset

from open_clip.tokenizer import HFTokenizer


class CounterAnimal(VisionDataset):    
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
        self.root = os.path.join(root, "counteranimal", "LAION-final")
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
        with open(os.path.join(self.root, 'counter_animal_ds.pkl'), 'rb') as fp:
           dict_all = pickle.load(fp)
           
        ds = dict_all["train_ds"] if self.train else dict_all["test_ds"]
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
            
        text =  f"a photo of a {self.class_names[target]}"
        if self.tokenizer:
            text = self.tokenizer(text)[0]
            
        return img, text

    def __len__(self) -> int:
        return len(self.samples)
    
    
class CounterAnimal_labeled(CounterAnimal):
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