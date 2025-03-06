import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

from torch import Tensor

from torchvision.datasets.folder import  default_loader, is_image_file
from torchvision.datasets.vision import VisionDataset

from open_clip.tokenizer import HFTokenizer


class ToyotaSmartHomeImages(VisionDataset):
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
        train: bool = True,
        transform: Optional[Callable] = None,
        tokenizer: Optional[HFTokenizer] = None,
        actions_list: List[str] = [],
        few_shots: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(root, transform=transform)
        assert few_shots >= 0, f"few_shots argument should be a non-negative integer but {few_shots} was given"
        assert seed >= 0, f"few_shots argument should be a non-negative integer but {few_shots} was given"
        self.train = train
        self.tokenizer = tokenizer
        self.few_shots = few_shots
        self.seed = seed
        assert all(act_name in self.action_names for act_name in actions_list), f"All the elements of `actions_list` should be in `action_names`!"
        self.actions_list = actions_list
        self.samples = self.make_dataset()

    def make_dataset(self) -> List[Dict[str, str]]:
        """
        Generates a list of samples of a form (path_to_sample).
        """
        
        if self.train:
            subjects_split = list(self.training_subjects)
        else:
            subjects_split = list(self.all_subjects - self.training_subjects)
            
        with open(os.path.join(self.root, 'images_text_description_short_trunc.pkl'), 'rb') as fp:
           dict_captions = pickle.load(fp)
           
        set_descr = set()
        for fname in dict_captions.keys():
            set_descr.add(dict_captions[fname])
            
        self.descr2class = {descr: idx for idx, descr in enumerate(set_descr)}  
        self.class2descr = {str(idx): descr for idx, descr in enumerate(set_descr)}  
        instances = []
        all_image_names = [fname for fname in os.listdir(os.path.join(self.root, "images_trunc")) if fname.endswith('.jpg')]
        for fname in all_image_names:
                image_path = os.path.join(self.root, "images_trunc", fname)
                fname_split = fname.split("_")
                action = fname_split[0]
                subject = fname_split[1]
                if is_image_file(image_path) and subject in subjects_split and (action in self.actions_list or len(self.actions_list) == 0):
                    dict_sample = {"img_path": image_path, "caption": dict_captions[fname]}
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
                    samples_rand = [{"img_path": image_path, "caption": action} for image_path in sampled_list]
                    few_shot_instances.extend(samples_rand)
                    random.seed(self.seed + 2 * seed_num)
                    
            return few_shot_instances

        return instances

    def __getitem__(self, index: int) -> Tuple[Tensor, Union[Tensor, List[Tensor]]]:
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
        text = self.tokenizer(sample["caption"])[0]
            
        return img, text

    def __len__(self) -> int:
        return len(self.samples)
    
    
class ToyotaSmartHomeImages_labeled(ToyotaSmartHomeImages):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        few_shots: int = 0,
        actions_list: List[str] = [],
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, few_shots=few_shots, actions_list=actions_list)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index]
        img = default_loader(sample["img_path"])
        caption = sample["caption"]
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.descr2class[caption]
    
"""
class ToyotaSmartHomeImages(VisionDataset):
    all_subjects = {"p02", "p03", "p04", "p06", "p07", "p09", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p25"}
    training_subjects = {"p03", "p04", "p06", "p07", "p09", "p12", "p13", "p15", "p17", "p19", "p25"}
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        tokenizer: Optional[HFTokenizer] = None,
        full_descr: bool = False, #  Wheter to use the full description
    ) -> None:
        super().__init__(root, transform=transform)
        self.train = train
        self.tokenizer = tokenizer
        self.full_descr = full_descr
        self.samples = self.make_dataset()

    def make_dataset(self) -> List[Dict[str, str]]:
    # 
        Generates a list of samples of a form (path_to_sample).
    # 
        fname_descr = 'images_text_description.pkl' if self.full_descr else 'images_text_description_short.pkl'
        with open(os.path.join(self.root, fname_descr), 'rb') as fp:
           dict_captions = pickle.load(fp)
           
        set_descr_1 = set()
        if self.full_descr:
            set_descr_2 = set()
            for fname in dict_captions.keys():
                caption = dict_captions[fname]
                set_descr_1.add(caption[0])
                set_descr_2.add(caption[1])
            self.descr2class_2 = {descr: idx for idx, descr in enumerate(set_descr_2)}
        else:
            for fname in dict_captions.keys():
                set_descr_1.add(dict_captions[fname])
            
        self.descr2class_1 = {descr: idx for idx, descr in enumerate(set_descr_1)}
    
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, IMG_EXTENSIONS)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)
       
        instances = []
        all_images_name = [fname for fname in os.listdir(os.path.join(self.root, "images")) if fname.endswith('.jpg')]
        if self.full_descr:
            subjects_split = list(self.training_subjects) if self.train else list(self.all_subjects - self.training_subjects)
            for fname in all_images_name:
                image_path = os.path.join(self.root, "images", fname)
                subject = fname.split("_")[1]
                if is_valid_file(image_path) and subject in subjects_split:
                    dict_sample = {"img_path": image_path, "caption": dict_captions[fname]}
                    instances.append(dict_sample)
        else:
            all_videos_name = list(set([fname.split("_sec")[0] for fname in all_images_name]))
            for vname in all_videos_name:
                frames_name_list = [img for img in all_images_name if vname in img]
                test_fname = frames_name_list.pop(2)
                
                if self.train:
                    for fname in frames_name_list:
                        image_path = os.path.join(self.root, "images", fname)
                        if is_valid_file(image_path):
                            dict_sample = {"img_path": image_path, "caption": dict_captions[fname]}
                            instances.append(dict_sample)
                else:
                    image_path = os.path.join(self.root, "images", test_fname)
                    if is_valid_file(image_path):
                        dict_sample = {"img_path": image_path, "caption": dict_captions[test_fname]}
                        instances.append(dict_sample)
                        
        return instances

    def __getitem__(self, index: int) -> Tuple[Tensor, Union[Tensor, List[Tensor]]]:
        #
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        #
        sample = self.samples[index]
        img = default_loader(sample["img_path"])
        if self.transform is not None:
            img = self.transform(img)
        
        if self.full_descr:
            if self.train:
                text = self.tokenizer(sample["caption"][np.random.randint(2)])[0]
            else:
                text = [self.tokenizer(sample["caption"][0])[0], self.tokenizer(sample["caption"][1])[0]]
        else:
            text = self.tokenizer(sample["caption"])[0]
            
        return img, text

    def __len__(self) -> int:
        return len(self.samples)
    
    
class ToyotaSmartHomeImages_labeled(ToyotaSmartHomeImages):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        full_descr: bool = False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, full_descr=full_descr)
        
    def __getitem__(self, index: int):
        #
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        #
        sample = self.samples[index]
        img = default_loader(sample["img_path"])
        caption = sample["caption"]
        if self.transform is not None:
            img = self.transform(img)
        
        if self.full_descr:
            label_1 = self.descr2class_1[caption[0]]
            label_2 = self.descr2class_2[caption[1]]
            return img, label_1, label_2
        else:       
            return img, self.descr2class_1[caption]
"""