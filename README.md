# Code used for the experiments in the paper "[Efficient Few-Shot Continual Learning in Vision-Language Models](https://arxiv.org/pdf/2502.04098)"

## Environment Requirements

Ensure that all the dependencies have been installed using the `requirements.txt` file.

## Data preparation
All datasets are publicly available. They should be downloaded at a pre-determined directory (say </my/directory/datasets>) and each folder should be named as ["tsi", "gtsrb", "aircraft", "counteranimal", "vsr", "hm", "eurosat", "mmvp", "visonlyqa"]. The dataset directory should be specified by the argument `--dataroot` when we train the model using [train.py](https://github.com/aresPanos/LoRSU/blob/main/src/train.py) 

## Fine-tune the CLIP-L-14 vision encoder using LoRSU and CLIP loss
We fine-tune the CLIP-L-14 vision encoder using LoRSU and CLIP loss rank=64, top-2 attention heads for the CL-5 setting.
    python src/train.py --ft_method lorsu_v_clip --lorsu_rank 64 --lorsu_alpha 64 --sparsity 0.1 --top_k_heads 2 --num_train_epochs 5 --dataset tsi --dataroot </my/directory/datasets> --learning_rate 1e-5 --weight_decay 0.01 --dataloader_num_workers 6 --is_cl True --few_shots 5 --per_device_train_batch_size 16 --log_folder </my/directory/logs> --checkpoint_folder </my/directory/checkpoints>

The fine-tuned vision encoder will be stored at  `</my/directory/checkpoints/CLIP-loss/CL/vsr/FS-5/LoRSU/rank-64_alpha-64/topk-heads-2/sparsity-100/epochs-5_batch-16_lr-000010/checkpoint_session-<session>_epoch-<epoch>.pt`


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/aresPanos/dtpp/blob/main/LICENSE) file for details.

## Acknowledgements
Our code is based on the [LLaVA repository](https://github.com/haotian-liu/LLaVA)
