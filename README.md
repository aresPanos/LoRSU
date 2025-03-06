# Code used for the experiments in the paper "[Efficient Few-Shot Continual Learning in Vision-Language Models](https://arxiv.org/pdf/2502.04098)"

## Environment Requirements

Ensure that all the dependencies have been installed using the `requirements.txt` file.

## Data preparation
All datasets are publicly available. They should be downloaded at a pre-determined directory (say </my/directory/datasets>) and each folder should be named as ["tsi", "gtsrb", "aircraft", "counteranimal", "vsr", "hm", "eurosat", "mmvp", "visonlyqa"]. The dataset directory should be specified by the argument `--dataroot` when we train the model using [train.py](https://github.com/aresPanos/LoRSU/blob/main/src/train.py) 

## Fine-tune the CLIP-L-14 vision encoder using LoRSU and CLIP loss

We train and evaluate the inter-event time model and the mark model, separately. 

To train and evaluate the time model using two mixture components on Taxi dataset, run

    python src/train.py --ft_method lorsu_v_clip --lorsu_rank 64 --lorsu_alpha 64 --sparsity 0.1 --top_k_heads 2 --num_train_epochs 5 --dataset tsi --dataroot </my/directory/datasets> --learning_rate 1e-5 --weight_decay 0.01 --dataloader_num_workers 6 --is_cl True --few_shots $num_shots --per_device_train_batch_size 16

The learned parameters of the mixture of log-Normals are stored at  `<directory_of_logs>/taxi/time_dist/saved_models/model_numMixtures-2.pt`


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/aresPanos/dtpp/blob/main/LICENSE) file for details.

## Acknowledgements
Our code is based on the [LLaVA repository](https://github.com/haotian-liu/LLaVA)
