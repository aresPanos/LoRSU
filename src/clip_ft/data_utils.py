import os
from typing import Tuple, List, Dict, Optional

import numpy as np

from torch.utils.data import DataLoader

from torchvision.datasets.vision import VisionDataset

from open_clip import get_tokenizer

from clip_ft.datasets.dalle import Dalle_labeled
from clip_ft.datasets.toyota_smarthome import ToyotaSmartHomeImages_labeled, ToyotaSmartHomeImages
from clip_ft.datasets.gtsrb import GTSRB_labeled, GTSRB
from clip_ft.datasets.aircraft import Aircraft_labeled, Aircraft
from clip_ft.datasets.eurosat import EuroSAT_labeled, EuroSAT
from clip_ft.datasets.counteranimal import CounterAnimal, CounterAnimal_labeled
from clip_ft.datasets.vqa_datasets import VSRTrain, HMTrain, VisOnlyTrain

from llava.train.train_utils import DataArguments, TrainingArguments

from clip_ft.transform import image_transform_clip, PreprocessCfg


def get_dataloader(training_args: TrainingArguments, dataset: VisionDataset, is_train: bool = False) -> DataLoader:
    bsize = training_args.train_batch_size if is_train else training_args.eval_batch_size
    return DataLoader(dataset, batch_size=bsize, shuffle=is_train, 
                      num_workers=training_args.dataloader_num_workers, drop_last=(is_train and len(dataset) > bsize))
    
    
def get_dalle_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg) -> DataLoader:
    dalle_dir = os.path.join(data_args.dataroot, 'dalle')        
    img_transf = image_transform_clip(False, cfg)
    dalle_val = Dalle_labeled(root=dalle_dir, transform=img_transf)
    dl = get_dataloader(training_args, dalle_val, False)
    
    return dl


def get_toyotahome_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg, train: bool = False, labeled: bool = True) -> DataLoader:
    toyotahome_dir = os.path.join(data_args.dataroot, 'toyota_smarthome_images')        
    img_transf = image_transform_clip(train, cfg)
    fshots = data_args.few_shots if train else 0        
    if labeled:
        toyotahome_ds = ToyotaSmartHomeImages_labeled(root=toyotahome_dir, train=train, transform=img_transf, few_shots=fshots, actions_list=[])
    else:
        tknzr = get_tokenizer('ViT-L-14-336')
        toyotahome_ds = ToyotaSmartHomeImages(root=toyotahome_dir, train=train, 
                                              transform=img_transf, tokenizer=tknzr, few_shots=fshots, actions_list=[])
        
    dl = get_dataloader(training_args, toyotahome_ds, train)
    
    return dl


def get_gtsrb_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg, train: bool = False, labeled: bool = True) -> DataLoader:   
    img_transf = image_transform_clip(train, cfg)
    fshots = data_args.few_shots if train else 0
    if labeled:
        gstrb_ds = GTSRB_labeled(root=data_args.dataroot, train=train, transform=img_transf, few_shots=fshots)
    else:
        tknzr = get_tokenizer('ViT-L-14-336') if cfg else get_tokenizer('EVA01-g-14')
        gstrb_ds = GTSRB(root=data_args.dataroot, train=train, transform=img_transf, tokenizer=tknzr, few_shots=fshots)
    dl = get_dataloader(training_args, gstrb_ds, train)
    
    return dl


def get_aircraft_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg, train: bool = False, labeled: bool = True) -> DataLoader:   
    aircraft_dir = os.path.join(data_args.dataroot, "fgvc_aircraft")
    img_transf = image_transform_clip(train, cfg)
    fshots = data_args.few_shots if train else 0
    if labeled:
        aircraft_ds = Aircraft_labeled(root=aircraft_dir, train=train, transform=img_transf, few_shots=fshots)
    else:
        tknzr = get_tokenizer('ViT-L-14-336')
        aircraft_ds = Aircraft(root=aircraft_dir, train=train, transform=img_transf, tokenizer=tknzr, few_shots=fshots)
    dl = get_dataloader(training_args, aircraft_ds, train)
    
    return dl


def get_eurosat_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg, train: bool = False, labeled: bool = True) -> DataLoader:   
    img_transf = image_transform_clip(train, cfg)
    fshots = data_args.few_shots if train else 0
    if labeled:
        eurosat_ds = EuroSAT_labeled(root=data_args.dataroot, train=train, transform=img_transf, few_shots=fshots)
    else:
        tknzr = get_tokenizer('ViT-L-14-336') if cfg else get_tokenizer('EVA01-g-14')
        eurosat_ds = EuroSAT(root=data_args.dataroot, train=train, transform=img_transf, tokenizer=tknzr, few_shots=fshots)
    dl = get_dataloader(training_args, eurosat_ds, train)
    
    return dl


def get_counteranimal_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg, train: bool = False, labeled: bool = True) -> DataLoader:   
    img_transf = image_transform_clip(train, cfg)
    fshots = data_args.few_shots if train else 0
    if labeled:
        counteranimal_ds = CounterAnimal_labeled(root=data_args.dataroot, train=train, transform=img_transf, few_shots=fshots)
    else:
        tknzr = get_tokenizer('ViT-L-14-336') if cfg else get_tokenizer('EVA01-g-14')
        counteranimal_ds = CounterAnimal(root=data_args.dataroot, train=train, transform=img_transf, tokenizer=tknzr, few_shots=fshots)
    dl = get_dataloader(training_args, counteranimal_ds, train)
    
    return dl


def get_vsr_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg) -> DataLoader:   
    img_transf = image_transform_clip(True, cfg)
    tknzr = get_tokenizer('ViT-L-14-336')
    vqa_ds = VSRTrain(root=data_args.dataroot, transform=img_transf, tokenizer=tknzr)
    
    dl = get_dataloader(training_args, vqa_ds, True)
    
    return dl


def get_hm_dl(data_args: DataArguments, training_args: TrainingArguments, cfg: Optional[PreprocessCfg] = None) -> DataLoader:   
    img_transf = image_transform_clip(True, cfg)
    tknzr = get_tokenizer('ViT-L-14-336')
    vqa_ds = HMTrain(root=data_args.dataroot, transform=img_transf, tokenizer=tknzr)
    
    dl = get_dataloader(training_args, vqa_ds, True)
    
    return dl


def get_cl_dataloaders(data_args: DataArguments, training_args: TrainingArguments, cfg: PreprocessCfg) -> List[DataLoader]:
    tknzr = get_tokenizer('ViT-L-14-336')
    list_train_dl = []
    img_transf_train = image_transform_clip(True, cfg)
    
    if data_args.dataset == "tsi":
        toyotahome_dir = os.path.join(data_args.dataroot, 'toyota_smarthome_images')  
        list_actions_per_session = [['WatchTV', 'Laydown', 'Sitdown', 'Pour.Fromkettle', 'Enter', 'Drink.Frombottle'],
                                    ['Eat.Attable', 'Pour.Frombottle', 'Cook.Cleandishes', 'Maketea.Boilwater', 'Leave', 'Cook.Cleanup'],
                                    ['Maketea.Insertteabag', 'Makecoffee.Pourwater', 'Drink.Fromcan', 'Readbook', 'Cutbread'],
                                    ['Drink.Fromcup', 'Drink.Fromglass', 'Usetablet', 'Pour.Fromcan', 'Usetelephone'],
                                    ['Walk', 'Cook.Stir', 'Makecoffee.Pourgrains', 'Cook.Cut', 'Uselaptop'],
                                   ]
        for session_actions_list in list_actions_per_session:
            toyotahome_ds_train = ToyotaSmartHomeImages(root=toyotahome_dir, 
                                                        train=True, 
                                                        transform=img_transf_train, 
                                                        tokenizer=tknzr, 
                                                        few_shots=data_args.few_shots, 
                                                        seed=training_args.seed,
                                                        actions_list=session_actions_list)
            
            
            dl_train = get_dataloader(training_args, toyotahome_ds_train, True)
            list_train_dl.append(dl_train)   
    elif data_args.dataset == "vsr":
        list_objects_per_session = [['oven', 'dining table', 'spoon', 'boat', 'cake', 'donut', 'sandwich'],
                                    ['fire hydrant', 'elephant', 'airplane', 'truck', 'apple', 'hot dog', 'sheep'],
                                    ['kite', 'baseball glove', 'cow', 'tie', 'scissors', 'toaster', 'tv'],
                                    ['bicycle', 'banana', 'couch', 'teddy bear', 'bus', 'umbrella', 'bird'],
                                    ['potted plant', 'bowl', 'broccoli', 'bottle', 'knife', 'orange', 'person', 'pizza'],
                                   ]
        for session_objects_list in list_objects_per_session:
            vsr_ds_train = VSRTrain(root=data_args.dataroot, 
                                    transform=img_transf_train, 
                                    tokenizer=tknzr, 
                                    objects_list=session_objects_list,
                                    few_shots=data_args.few_shots, 
                                    seed=training_args.seed)
            
            dl_train = get_dataloader(training_args, vsr_ds_train, True)
            list_train_dl.append(dl_train) 
    elif data_args.dataset == "hm":        
        for session_num in range(data_args.num_sessions):
            vsr_ds_train = HMTrain(root=data_args.dataroot, 
                                   transform=img_transf_train, 
                                   tokenizer=tknzr, 
                                   session=session_num,
                                   few_shots=data_args.few_shots, 
                                   seed=training_args.seed)
            
            dl_train = get_dataloader(training_args, vsr_ds_train, True)
            list_train_dl.append(dl_train)
    elif data_args.dataset == "visonlyqa":
        for session_num in range(data_args.num_sessions):
            visonly_ds_train = VisOnlyTrain(root=data_args.dataroot, 
                                            transform=img_transf_train, 
                                            tokenizer=tknzr, 
                                            session=session_num,
                                            few_shots=data_args.few_shots, 
                                            seed=training_args.seed)
            
            dl_train = get_dataloader(training_args, visonly_ds_train, True)
            list_train_dl.append(dl_train)
    elif data_args.dataset == "aircraft":
        aircraft_dir = os.path.join(data_args.dataroot, "fgvc_aircraft")
        classes = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 
                   19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 
                   84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0,  72, 35, 
                   58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 
                   32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
        session_classes_list = np.array_split(classes, data_args.num_sessions)
        for classes_session in session_classes_list:
            aircraft_ds_train = Aircraft(root=aircraft_dir, 
                                         train=True, 
                                         transform=img_transf_train, 
                                         tokenizer=tknzr, 
                                         few_shots=data_args.few_shots, 
                                         seed=training_args.seed,
                                         classes_session=classes_session)
            
            dl_train = get_dataloader(training_args, aircraft_ds_train, True)
            list_train_dl.append(dl_train)
    elif data_args.dataset == "eurosat":
        classes = [0, 1 , 2, 3, 4, 5, 6, 7, 8, 9]
        session_classes_list = np.array_split(classes, data_args.num_sessions)
        for classes_session in session_classes_list:
            eurosat_ds_train = EuroSAT(
                root=data_args.dataroot, 
                train=True, 
                transform=img_transf_train, 
                tokenizer=tknzr, 
                few_shots=data_args.few_shots, 
                seed=training_args.seed,
                classes_session=classes_session)
            
            dl_train = get_dataloader(training_args, eurosat_ds_train, True)
            list_train_dl.append(dl_train)
    elif data_args.dataset == "counteranimal":
        classes = [150, 296, 79, 57, 144, 133, 349, 293, 100, 
                   49, 305, 128, 54, 80, 16, 275, 76, 360, 
                   30, 316, 33, 39, 71, 89, 37, 23, 42, 
                   337, 81, 357, 9, 290, 41, 70, 102, 279, 
                   130, 291, 20, 10, 56, 277, 276, 58, 83]
        session_classes_list = np.array_split(classes, data_args.num_sessions)
        for classes_session in session_classes_list:
            counteranimal_ds_train = CounterAnimal(root=data_args.dataroot, 
                                                   train=True, 
                                                   transform=img_transf_train, 
                                                   tokenizer=tknzr, 
                                                   classes_session=classes_session,
                                                   few_shots=data_args.few_shots, 
                                                   seed=training_args.seed)            
            
            dl_train = get_dataloader(training_args, counteranimal_ds_train, True)
            list_train_dl.append(dl_train)
    else:
        classes = [25, 2, 11, 1, 40, 27, 5, 9, 17, 32, 29, 20, 39, 21, 15, 23, 10, 3, 18, 38,
                   42, 14, 22, 35, 34, 19, 33, 12, 26, 41, 0, 37, 6, 13, 24, 30, 28, 31, 7, 16, 4, 36, 8]
        session_classes_list = np.array_split(classes, data_args.num_sessions)
        for classes_session in session_classes_list:
            gtsrb_ds_train = GTSRB(root=data_args.dataroot, 
                                   train=True, 
                                   transform=img_transf_train, 
                                   tokenizer=tknzr, 
                                   few_shots=data_args.few_shots, 
                                   seed=training_args.seed,
                                   classes_session=classes_session)
            
            dl_train = get_dataloader(training_args, gtsrb_ds_train, True)
            list_train_dl.append(dl_train)
        
    return list_train_dl
