import os
import math
import copy
from typing import Any, Dict, List, Optional, Literal, Union, Tuple, Callable
import datetime
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
from loguru._logger import Logger

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from peft import PeftModel, LoraConfig, get_peft_model

from PIL import Image

import transformers
from transformers.tokenization_utils import PreTrainedTokenizer

from llava.model import LlavaLlamaForCausalLM
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.train.train import preprocess, preprocess_multimodal, DataCollatorForSupervisedDataset
from llava.train.lora_utils import MaskedLoRALinear, LoRALinear, get_submodules, replace_module
from llava.datasets.vqa_datasets import ToyotaSmartHomeImagesVQAFT, GTSRBVQAFT, FGVCAircraftVQAFT, EuroSATVQAFT, \
                                        CounterAnimalVQAFT, HMTrainFT, VSRFT, MMVP_VQA_FT, Vis_Only_QAFT
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llava-v1.5-7b-lora")
    model_path: Optional[str] = field(default="/home/ap2313/rds/hpc-work/code/LLaVA/llava_models/llava-v1.5-7b-lora")
    model_base: Optional[str] = field(default="/home/ap2313/rds/hpc-work/code/LLaVA/llava_models/vicuna-7b-v1-5")
    model_name: Optional[str] = field(default="llava-v1.5-7b-lora")
    fine_tuned_clip_model: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1.5")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    sep: Optional[str] = field(default=",")
    num_params_init: Optional[int] = field(default=0)
    device: Optional[Union[str, torch.device]] = field(default=None)
    num_beams: int = 1
    temperature: float = 0.0
    top_p: Optional[float] = field(default=None)
    max_new_tokens: int = 20
    conv_mode: Literal["v0", "v1", "vicuna_v1", "llama_2", "mistral_instruct",\
                       "chatml_direct", "mistral_direct", "plain", "v0_plain", \
                       "llava_v0", "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"] = 'vicuna_v1'


@dataclass
class DataArguments:
    dataroot: str = field(default="/home/ap2313/rds/hpc-work/datasets",
                           metadata={"help": "Path to the training data."})
    dataset: Literal["tsi", "gtsrb", "aircraft", "counteranimal", "vsr", "hm", "eurosat", "mmvp", "visonlyqa"] = 'tsi'
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_aspect_ratio: str = 'pad'
    is_cl: bool = False
    is_cl_5d: bool = False
    is_cl_5d_full: bool = False
    num_sessions: int = 5
    few_shots: Optional[int] = field(default=0)
    image_processor: Optional[transformers.CLIPImageProcessor] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)    
    optim: str = field(default="adamw_torch")
    
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    ckpt_fname: Optional[str] = field(default="")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    precision: Literal['fp32', 'fp16', 'bf16'] = 'fp16'
    ft_method: Literal['lorsu_v_clip', 'lora_v_clip', 'lorsu_llm', 'lora_llm', 'lorsu_v_ppl', 'lora_v_ppl', 'full_lora', 'lora_lastlayers'] = 'lorsu_v_clip'
    grad_clip_norm: Optional[float] = field(default=None)
    save_frequency: int = 1
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=True)
    lorsu_rank: int = 8
    lorsu_alpha: int = 8
    top_k_heads: int = 8
    sparsity: float = 0.01
    grad_total_points_mask: int = 1000
    train_last_k_layers: int = 16
    output_dir: Optional[str] = field(default="./output", metadata={"help": "Directory of the log files."})
    log_folder: Optional[str] = field(default="/home/ap2313/rds/hpc-work/code/code-llm/logs", metadata={"help": "Directory of the log files."})
    checkpoint_folder: Optional[str] = field(default="/home/ap2313/rds/hpc-work/code/code-llm/checkpoints", metadata={"help": "Directory of the checkpoint files."})
    log_freq: int = 50
    eval_batch_size: int = 64
    seed: Optional[int] = field(default=2024) 
    lr_decay_rate: Optional[float] = field(default=None)
    min_lr: Optional[float] = field(default=1e-6)
    warmup_lr: Optional[float] = field(default=1e-6)
        
    
def parsed_args_train() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args.device = get_device()
    
    assert data_args.few_shots >= 0., f"argument `few_shots` should be a non-negative integer but {data_args.few_shots} was given!"
    
    if training_args.ft_method in ["full_lora", "lora_lastlayers"]:
        training_args.lora_enable = True
    
    if training_args.ft_method in ["lora_lorsu", "lora_lorsu_v", "full_lora", "lora_lastlayers"]:
        assert training_args.lora_rank > 0., f"argument `lora_rank` should be a positive integer but {training_args.lora_rank} was given!"
        assert training_args.lora_alpha > 0., f"argument `lora_alpha` should be a positive number but {training_args.lora_alpha} was given!"
        
        assert training_args.lorsu_rank > 0., f"argument `lorsu_rank` should be a positive integer but {training_args.lorsu_rank} was given!"
        assert training_args.lorsu_alpha > 0., f"argument `lorsu_alpha` should be a positive number but {training_args.lorsu_alpha} was given!"
        assert training_args.lora_enable or training_args.lorsu_enable, f"Either one of the arguments `lora_enable` or `lorsu_enable` should be set to `True`"
                
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end

    return model_args, data_args, training_args


def set_seed(seed: int):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enable PyTorch deterministic mode. This potentially requires either the environment
        # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
        # depending on the CUDA version, so we set them both here
        
        #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        #torch.use_deterministic_algorithms(True, warn_only=warn_only)
        
        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
                  
        
def get_device() -> torch.device:
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(conv.roles[0], text) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


def apply_to_sample(f: Callable[[torch.Tensor], torch.Tensor], sample: Dict[str, torch.Tensor]):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample: Dict[str, torch.Tensor]):
    def _move_to_cuda(tensor: torch.Tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], cuda_enabled=True) -> Dict[str, Union[torch.Tensor, Any]]:
    if cuda_enabled:
        inputs = move_to_cuda(inputs)
    return inputs

def calculate_model_size_GB(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_gb = (param_size + buffer_size) / 1024**3
    return size_all_gb

def float2str(number: float, digits: Optional[int] = None) -> str:
    float_str = f"{number:.6f}"
    decimals = float_str.split(".")[1]
    if digits:
        assert digits > 0 and digits < 7, "Invalid digits value. The value must be an integer between 1 and 6 but {digits} was given"
        decimals = decimals[:digits]
    return decimals


def create_logname_train(data_args: DataArguments, training_args: TrainingArguments) -> str:
    assert os.path.isdir(training_args.log_folder), f"`{training_args.log_folder}` is not valid directory."    
    date_format = "%Y%m%d-%H%M%S"
    now_timestamp = datetime.datetime.now().strftime(date_format)
    setting = "CL" if data_args.is_cl else "offline"
    fs_num = f"FS-{data_args.few_shots}" if data_args.is_cl else ""
    train_mode = os.path.join(setting, data_args.dataset, fs_num)

    if 'clip' in training_args.ft_method:
        folder_clip_llm = "CLIP-loss"               
        if 'lorsu_v_clip' == training_args.ft_method:
            train_folder = os.path.join(f"LoRSU", f"rank-{training_args.lorsu_rank}_alpha-{int(training_args.lorsu_alpha)}")
            sparsity_str = float2str(training_args.sparsity, 3)
            train_folder = os.path.join(train_folder, f"topk-heads-{training_args.top_k_heads}", f"sparsity-{sparsity_str}")
        else:
            train_folder = os.path.join("LoRA", f"rank-{training_args.lora_rank}_alpha-{int(training_args.lora_alpha)}")
    else:
        folder_clip_llm = "Perplexity-loss"
        mm_proj = "+mm_projector" if training_args.mm_projector_lr else ""

        suffix = "-V" if "_v" in training_args.ft_method else ""
        if 'lora' in training_args.ft_method:
            lora_method = "LoRA-Full" if "full" in training_args.ft_method else "LoRA"
            if training_args.ft_method == "lora_lastlayers":
                lora_method += f"-{training_args.train_last_k_layers}"
            train_folder = os.path.join(lora_method + suffix, f"rank-{training_args.lora_rank}_alpha-{int(training_args.lora_alpha)}" + mm_proj)
        else:
            suffix = suffix if "_v" in training_args.ft_method else f"-{training_args.train_last_k_layers}"
            train_folder = os.path.join(f"LoRSU" + suffix, f"rank-{training_args.lorsu_rank}_alpha-{int(training_args.lorsu_alpha)}")
            sparsity_str = float2str(training_args.sparsity, 3)
            train_folder = os.path.join(train_folder, f"topk-heads-{training_args.top_k_heads}" + mm_proj, f"sparsity-{sparsity_str}")

    lr_str = float2str(training_args.learning_rate)
    train_folder = os.path.join(train_folder, f"epochs-{int(training_args.num_train_epochs)}_batch-{training_args.train_batch_size}_lr-{lr_str}")        
    log_dir = os.path.join(training_args.log_folder, folder_clip_llm, train_mode, train_folder)
    os.makedirs(log_dir, exist_ok=True)
    log_fname = os.path.join(log_dir, f"log_train_{now_timestamp}.txt")
    
    if 'clip' in training_args.ft_method:
        checkpoint_path = os.path.join(training_args.checkpoint_folder, folder_clip_llm, train_mode, train_folder)
        os.makedirs(checkpoint_path, exist_ok=True)
        training_args.ckpt_fname = os.path.join(checkpoint_path, "checkpoint_session-{}_epoch-{}.pt") if data_args.is_cl else os.path.join(checkpoint_path, "checkpoint_epoch-{}.pt")
    
    return log_fname
    

def create_data_dict(ds_name: str, 
                     dataroot: str, 
                     classes_session: Optional[Union[List[int], List[str], int]]= None, 
                     few_shots: int = 0, #
                     seed: int = 0) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    assert ds_name in ["tsi", "gtsrb", "aircraft", "counteranimal", "vsr", "hm", "eurosat", "mmvp", "visonlyqa"], f"Invalid dataset name: `{ds_name}`"
    
    if ds_name == "tsi":
        actions_list = classes_session
        if actions_list is None:
            actions_list = []
        dataset = ToyotaSmartHomeImagesVQAFT(root=dataroot, actions_list=actions_list, few_shots=few_shots, seed=seed)
    if ds_name == "vsr":
        dataset = VSRFT(root=dataroot, objects_list=classes_session, few_shots=few_shots, seed=seed)
    if ds_name == "hm":
        classes_session = -1 if classes_session is None else classes_session
        dataset = HMTrainFT(root=dataroot, session=classes_session, few_shots=few_shots, seed=seed)
    elif ds_name == "gtsrb":
        dataset = GTSRBVQAFT(root=dataroot, classes_session=classes_session, few_shots=few_shots, seed=seed)
    elif ds_name == "aircraft":
        dataset = FGVCAircraftVQAFT(root=dataroot, classes_session=classes_session, few_shots=few_shots, seed=seed)
    elif ds_name == "eurosat":
        dataset = EuroSATVQAFT(root=dataroot, classes_session=classes_session, few_shots=few_shots, seed=seed)
    elif ds_name == "counteranimal":
        dataset = CounterAnimalVQAFT(root=dataroot, classes_session=classes_session, few_shots=few_shots, seed=seed)
    elif ds_name == "mmvp":
        classes_session = -1 if classes_session is None else classes_session
        dataset = MMVP_VQA_FT(root=dataroot, session=classes_session)
    elif ds_name == "visonlyqa":
        classes_session = -1 if classes_session is None else classes_session
        dataset = Vis_Only_QAFT(session=classes_session, few_shots=few_shots, seed=seed)
        
    if ds_name == "visonlyqa":
        string_question = DEFAULT_IMAGE_TOKEN + "\n{} "
    else:
        string_question = DEFAULT_IMAGE_TOKEN + "\nBased on the image, respond to this question with a short answer: {} "

    # Initialize list to hold all JSON data
    list_data_dict = []

    # Process and save images and labels
    for i in range(len(dataset)):
        img_path, question, answer = dataset[i]
        # Structure for LLaVA JSON
        json_data = {
            "id": i+1,
            "image": img_path,
            "conversations": [
                {
                    "from": "human",
                    #"value": f"{DEFAULT_IMAGE_TOKEN}\nBased on the image, respond to this question with a short answer: {question} "
                    "value": string_question.format(question)
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }


        # Append to list
        list_data_dict.append(json_data)
    return list_data_dict


def get_llm_target_modules(start_layer: int = 0):
    lora_module_names = list()
    num_layers = 32

    for layer in range(start_layer, num_layers):
        q_proj_name = f"model.layers.{layer}.self_attn.q_proj"
        k_proj_name = f"model.layers.{layer}.self_attn.k_proj"
        v_proj_name = f"model.layers.{layer}.self_attn.v_proj"
        o_proj_name = f"model.layers.{layer}.self_attn.o_proj"
        gate_proj_name = f"model.layers.{layer}.mlp.gate_proj"
        up_proj_name = f"model.layers.{layer}.mlp.up_proj"
        down_proj_name = f"model.layers.{layer}.mlp.down_proj"
        list_tmp = [q_proj_name, k_proj_name, v_proj_name, o_proj_name, gate_proj_name, up_proj_name, down_proj_name]
        
        #list_tmp = [q_proj_name, v_proj_name]
        lora_module_names.extend(list_tmp)

    return list(lora_module_names)


def get_vision_target_modules():
    num_layers = 23
    lora_module_names = list()
    prefix = "model.vision_tower.vision_tower.vision_model.encoder."
    for layer in range(num_layers):
        q_proj_name = prefix + f"layers.{layer}.self_attn.q_proj"
        k_proj_name = prefix + f"layers.{layer}.self_attn.k_proj"
        v_proj_name = prefix + f"layers.{layer}.self_attn.v_proj"
        out_proj_name = prefix + f"layers.{layer}.self_attn.out_proj"
        fc1_name = prefix + f"layers.{layer}.mlp.fc1"
        fc2_name = prefix + f"layers.{layer}.mlp.fc2"
        list_tmp = [q_proj_name, k_proj_name, v_proj_name, out_proj_name, fc1_name, fc2_name]

        lora_module_names.extend(list_tmp)

    return list(lora_module_names)


def full_target_modules():
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj', 'fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj', 'lm_head', 'mm_projector.0', 'mm_projector.2']


def get_model(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> Tuple[Union[LlavaLlamaForCausalLM, PeftModel], PreTrainedTokenizer]:
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_args.model_path, model_args.model_base, model_args.model_name)
    if 'lora' in training_args.ft_method:
        if training_args.ft_method == "full_lora":
            t_modules = full_target_modules()
        elif training_args.ft_method == 'lora_llm':
            t_modules = get_llm_target_modules()
        elif training_args.ft_method == 'lora_lastlayers':
            t_modules = get_llm_target_modules(training_args.train_last_k_layers)
        elif training_args.ft_method == 'lora_v_ppl':
            t_modules = get_vision_target_modules()
        else:
            raise ValueError(f"Not recogized fine-tuning method {training_args.ft_method}")
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=t_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.precision != "fp32":
            if training_args.precision == "bf16":
                model.to(torch.bfloat16)
            if training_args.precision == "fp16":
                model.to(torch.float16)
        model = get_peft_model(model, lora_config)
        
    num_params_init = sum(param.data.nelement() for name, param in model.named_parameters() if "vision_tower" not in name)        
    data_args.image_processor = image_processor
    data_args.data_seed = training_args.seed
    model_args.num_params_init = num_params_init   
  
    if training_args.ft_method != "full_lora":
        use_mm_projector = training_args.mm_projector_lr is not None
        if use_mm_projector:
            set_lora_to_mm_projector(model.get_model().mm_projector)
                        
        for name, param in model.get_model().mm_projector.named_parameters():
            param.requires_grad = use_mm_projector and any(nn in name for nn in ["lora_", "bias"]) 
            param.requires_grad = use_mm_projector and "lora_" in name
                
    return model, tokenizer


def get_nb_trainable_parameters(model: Union[LlavaLlamaForCausalLM, PeftModel]) -> Tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

def print_trainable_parameters(model: Union[LlavaLlamaForCausalLM, PeftModel], lggr: Logger) -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = get_nb_trainable_parameters(model)

        lggr.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )


class LazySupervisedDatasetVQAFT(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 data_args: DataArguments, 
                 classes_session: Optional[Union[List[int], List[str]]]= None):
        super(LazySupervisedDatasetVQAFT, self).__init__()
        list_data_dict = create_data_dict(ds_name=data_args.dataset, 
                                          dataroot=data_args.dataroot, 
                                          classes_session=classes_session,
                                          few_shots=data_args.few_shots,
                                          seed=data_args.data_seed) 

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            if isinstance(image_file, str):
                image = Image.open(image_file).convert('RGB')
            else:
                image = image_file
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
    
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments,
                                classes_session: Optional[Union[List[int], List[str]]]= None) -> Tuple[LazySupervisedDatasetVQAFT, DataCollatorForSupervisedDataset]:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDatasetVQAFT(tokenizer=tokenizer, data_args=data_args, classes_session=classes_session) 
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, data_collator
    
    
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        return
        
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
        

class MetricLoggerLoguru(object):
    def __init__(self, lggr: Logger, delimiter="\t"):
        self.lggr = lggr
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.1f} GB")
        log_msg = self.delimiter.join(log_msg)
        GB = 1024.0 ** 3
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.lggr.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / GB,
                        )
                    )
                else:
                    self.lggr.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.lggr.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )
        
        
@torch.no_grad()
def mask_model_gradients(model: LlavaLlamaForCausalLM, mask: Dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        if param.requires_grad and "mlp.up_proj" in name and "vision_tower" not in name:
            assert param.grad is not None, f"The gradient of parameter `{name}` is None!"
            assert param.grad.dtype == torch.float32, f"The gradient of parameter `{name}` should be float32 but its is {param.grad.dtype}"
            param.grad *= mask[name].float()  
            
@torch.no_grad()
def mask_model_gradients_vision_tower(model: LlavaLlamaForCausalLM, mask: Dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        if param.requires_grad and "mlp.fc1.weight" in name:
            assert param.grad is not None, f"The gradient of parameter `{name}` is None!"
            assert param.grad.dtype == torch.float32, f"The gradient of parameter `{name}` should be float32 but it is {param.grad.dtype}"
            param.grad *= mask[name].float()
            

def compute_importance_scores(model: LlavaLlamaForCausalLM, 
                              data_loader: DataLoader, 
                              num_samples_grad: int) -> Tuple[float, Dict[str, torch.Tensor]]:
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    iters_per_epoch = len(data_loader)
    importance_scores = {name: torch.zeros_like(param.data) for name, param in model.named_parameters() if param.requires_grad}
    if not hasattr(data_loader, "__next__"):
        # convert to iterator if not already
        data_loader = iter(data_loader)
            
    tic = time.time()
    num_samples_used = 0
    for _ in range(iters_per_epoch):
        inputs = next(data_loader)
        inputs = prepare_inputs(inputs, cuda_enabled=True)
        n_samples = inputs['images'].size(0)

        with torch.cuda.amp.autocast(enabled=True):
            loss = model(**inputs)["loss"]
            
        scaler.scale(loss).backward() 
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, training stopped!")
                
        num_samples_used += n_samples
        stop_flag = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise ValueError(f"The gradient of {name} is None!")

                importance_scores[name] += (n_samples * param.grad.data.clone())
                if importance_scores[name].abs().min() < 1e-10:
                    stop_flag = False
              
        if num_samples_used >= num_samples_grad and stop_flag:
            break
                           
    unbiased_scale = 1. / num_samples_used
    for name, param in model.named_parameters():
        if param.requires_grad:
            importance_scores[name] *= unbiased_scale 
    toc = time.time() - tic
    
    return toc, importance_scores

def set_lora_to_mm_projector(mm_projector: nn.Module):
    key_list = [k for k, _ in mm_projector.named_modules()]
    for key in key_list:       
            parent, target, target_name = get_submodules(mm_projector, key)
            if isinstance(target, torch.nn.Linear):
                bias = target.bias is not None
                kwargs = {
                    "r": 64,
                    "lora_alpha": 32,
                    "lora_dropout": 0.,
                    "fan_in_fan_out": False,
                    "merge_weights": True,
                    "bias": bias,
                }
                
                new_module = LoRALinear(target.in_features, target.out_features, **kwargs).cuda()
                replace_module(parent, target_name, new_module, target)  

def build_spu_attn_mask(model: LlavaLlamaForCausalLM, trainloader: DataLoader, training_args: TrainingArguments) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    num_layers = 32 if training_args.ft_method == "lora_lorsu" else 23
    start_layer = num_layers - training_args.train_last_k_layers if training_args.ft_method == "lora_lorsu" else 0
    prefix = "model" if training_args.ft_method == "lora_lorsu" else "model.vision_tower.vision_tower.vision_model.encoder" 
    
    if training_args.ft_method == "lora_lorsu":
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'up_proj']
    else:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'fc1']
        
    for name, param in model.named_parameters():
        param.requires_grad = any(sub_name in name for sub_name in target_modules) \
                              and any(f"{prefix}.layers.{layer}." in name for layer in range(start_layer, num_layers))  
                              
    target_modules.pop()   
    time_scores, importance_scores = compute_importance_scores(model, trainloader, training_args.grad_total_points_mask) 
    
    tic = time.time()  
        
    top_k_heads = training_args.top_k_heads
    sparsity = training_args.sparsity
    num_heads = 32 if training_args.ft_method == "lora_lorsu" else 16
    total_head_dim = 4096 if training_args.ft_method == "lora_lorsu" else 1024
    head_dim = total_head_dim // num_heads
    attn_trainable = 0
    mask_attn = {}
    mask_mlp_fc1_up = {}
    mlp_name = "up_proj" if training_args.ft_method == "lora_lorsu" else 'fc1'  
    
    for i in range(start_layer, num_layers):
        dq = importance_scores[f"{prefix}.layers.{i}.self_attn.q_proj.weight"].reshape(num_heads, head_dim, total_head_dim)
        dk = importance_scores[f"{prefix}.layers.{i}.self_attn.k_proj.weight"].reshape(num_heads, head_dim, total_head_dim)
        dv = importance_scores[f"{prefix}.layers.{i}.self_attn.v_proj.weight"].reshape(num_heads, head_dim, total_head_dim)
        
        score_attn = dq.square().sum(dim=(1, 2)) + dk.square().sum(dim=(1, 2)) + dv.square().sum(dim=(1, 2))
        _, topk_heads = torch.topk(score_attn, k=top_k_heads)  
            
        mask_msa_qkv = torch.zeros_like(dv, dtype=torch.bool)
        mask_msa_qkv[topk_heads] = True
        mask_msa_qkv = mask_msa_qkv.reshape(total_head_dim, total_head_dim)
        mask_attn[f"{prefix}.layers.{i}.self_attn.mask"] = mask_msa_qkv
        
        params_attn = min(mask_msa_qkv.sum().item(), training_args.lorsu_rank * 2 * total_head_dim)
                
        name_fc1_up = f"{prefix}.layers.{i}.mlp.{mlp_name}.weight"
        magnitudes_up = importance_scores[name_fc1_up].abs()
        top_k_grads_up = int(magnitudes_up.numel() * sparsity)
        _, topk_indices_up = torch.topk(magnitudes_up.view(-1), k=top_k_grads_up)
        mask_mlp_fc1_up[name_fc1_up] = torch.zeros_like(magnitudes_up, dtype=torch.bool)
        mask_mlp_fc1_up[name_fc1_up].view(-1)[topk_indices_up] = True
        params_f1_up = mask_mlp_fc1_up[name_fc1_up].sum().item()
        
        attn_trainable += (3 * params_attn + params_f1_up)
        
    key_list = [k for k, _ in model.named_modules()]
    for key in key_list:       
        target_module_found = any(key.endswith(target_key) for target_key in target_modules) \
                              and any(f"{prefix}.layers.{layer}." in key for layer in range(start_layer, num_layers)) 
        if target_module_found:
            parent, target, target_name = get_submodules(model, key)
            assert isinstance(target, torch.nn.Linear)
            bias = target.bias is not None
            name_qkv = ".".join(key.split(".")[:-1])
            mask_lora = mask_attn[name_qkv + '.mask']
            kwargs = {
                "r": training_args.lorsu_rank,
                "lora_alpha": training_args.lorsu_alpha,
                "lora_dropout": 0.,
                "fan_in_fan_out": False,
                "merge_weights": True,
            }
            
            new_module = MaskedLoRALinear(target.in_features, target.out_features, mask_lora, bias=bias, **kwargs).cuda()
            replace_module(parent, target_name, new_module, target)              
    toc = time.time() - tic            
      
    ft_mm_projector = training_args.mm_projector_lr is not None
    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name \
                               or any(f"{prefix}.layers.{layer}.mlp.{mlp_name}." in name for layer in range(start_layer, num_layers))
                               #\ or (ft_mm_projector and "bias" in name and "mm_projector" in name)    
    
        if ft_mm_projector and "mm_projector" in name and param.requires_grad:
            attn_trainable += param.numel()        
         
    return mask_mlp_fc1_up, toc, time_scores, attn_trainable
    
    
def merge_weights(model: LlavaLlamaForCausalLM, training_args: TrainingArguments):
    target_modules = ['q_proj', 'k_proj', 'v_proj'] 
    num_layers = 32 if training_args.ft_method == "lora_lorsu" else 23
    start_layer = num_layers - training_args.train_last_k_layers if training_args.ft_method == "lora_lorsu" else 0
    prefix = "model." if training_args.ft_method == "lora_lorsu" else "model.vision_tower.vision_tower.vision_model.encoder." 
    
    key_list = [k for k, _ in model.named_modules()]
    for key in key_list:        
        target_module_found = any(key.endswith(target_key) for target_key in target_modules) \
                              and any(prefix + f"layers.{layer}." in key for layer in range(start_layer, num_layers))
        if target_module_found:
            parent, target, target_name = get_submodules(model, key)
            assert isinstance(target, MaskedLoRALinear)
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias, device=target.weight.device)
            replace_module(parent, target_name, new_module, target)
            
            
class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
