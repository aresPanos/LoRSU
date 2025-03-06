import os
from typing import Any, Dict, List, Optional, Tuple, Callable

from torch import Tensor

from torchvision.datasets.folder import  default_loader, is_image_file
from torchvision.datasets.vision import VisionDataset


class Dalle_labeled(VisionDataset):
    class_names = ['stirring the pot', 'cutting bread', 'cleaning up', 'walking', 'eating', 
                   'using a tablet', 'cutting food', 'holding a cup', 'holding a glass', 
                   'sitting down', 'boiling water in a black kettle', 'holding a can', 
                   'watching TV', 'lying down', 'using a laptop', 'making tea', 'using a cordless phone', 
                   'using a white coffee machine', 'washing dishes', 'holding a black kettle', 'holding a bottle', 'reading a book']
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.samples = self.make_dataset()

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """                       
        self.descr2class = {f'a photo of a person {descr}': idx for idx, descr in enumerate(self.class_names)}  
        self.class2descr = {str(idx): f'a photo of a person {descr}' for idx, descr in enumerate(self.class_names)}  
        instances = []
        all_image_names = [fname for fname in os.listdir(os.path.join(self.root, "v2")) if fname.endswith('.png')]
        for fname in all_image_names:
                image_path = os.path.join(self.root, "v2", fname)
                if is_image_file(image_path):
                    fname_splt = fname.split("_")
                    action = fname_splt[0].replace("-", " ")
                    dict_sample = {"img_path": image_path, "label": self.descr2class[f'a photo of a person {action}']}
                    instances.append(dict_sample)
        return instances

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index]
        img = default_loader(sample["img_path"])
        if self.transform is not None:
            img = self.transform(img)
        
        return img, sample["label"]

    def __len__(self) -> int:
        return len(self.samples)
    
    