# Self-Supervised Skill Abstraction and Decision Making

## Model specific installations
0. ```conda with python 3.8```
1. ```pip install vector-quantize-pytorch==1.8.1```
2. ```pip install positional-encodings==6.0.3```
3. ```pip install einops==0.3.2```
4. ```pip install hydra-core==1.2.0```
5. Other dependencies like torch, torchvision, transformers, etc
6. Libero specific dependencies are in the `requirements.txt` file.

## Action Chunking Transformer
ACT Model files are in the `libero/lifelong/models` directory. Description:
1. [`act.py`](libero/lifelong/models/act.py): ACT policy file, uses base_policy class, imports submodules model and observation encoder.
2. [`modules/act_utils.py`](libero/lifelong/models/modules/act_utils.py): contains GPT model and other modules.
3. [`modules/rgb_modules.py`](libero/lifelong/models/modules/rgb_modules.py): contains visual observation encoder.
    - ACTResnetEncoder: Original Resnet encoder for visual observations.
    - EfficientACTResnetEncoder: Efficient Resnet encoder with token-learner.
4. [`quantize_utils.py`](libero/lifelong/quantize_utils.py): Contains quantization functions.

## Config Files
Config files are in the `libero/configs` directory. Description:
1. [`ACT.yaml`](libero/configs/ACT.yaml): ACT main config for training.
2. [`ACT_eval.yaml`](libero/configs/ACT_eval.yaml): ACT main config for evals. 

All of these files use default files in the `libero/configs` directory.  
Following are model specific config files:
1. [`policy/act_policy.yaml`](libero/configs/policy/act_policy.yaml): ACT model config.
2. [`policy/image_encoder/effact_resnet_encoder.yaml`](libero/configs/policy/image_encoder/effact_resnet_encoder.yaml): Token Learner config.

Read comments in the config files for more details.  

## Training Scripts
These are libero specific scripts that loads the model and data and environment (for evals) and trains the model.  
Main scripts for training are in the `libero/lifelong` directory. The main scripts are:
1. [`act_policy.py`](libero/lifelong/act_policy.py): Train the ACT model.
2. [`act_policy_eval.py`](libero/lifelong/act_policy_eval.py): Evaluation of the ACT policy.
These scripts load [`algo/multitask.py`](libero/lifelong/algos/multitask.py) algo that loads multitask data and trains the model. (libero specific)

1. Command to train ACT model:
```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/act_policy.py```
2. Command to evaluate:
```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/act_policy_eval.py```

You can change the `num_tok` parameter in the [`effact_resnet_encoder.yaml`](libero/configs/policy/image_encoder/effact_resnet_encoder.yaml) config file to change the number of tokens to learn with Token Learner.

## Running this on PACE

1. Start a free interactive session with a RTX6000 GPU: ```salloc -A gts-agarg35 -N1 --mem-per-gpu=12G -q embers -t8:00:00 --gres=gpu:RTX_6000:1```
2. Load in Conda: ```module load anaconda3/2022.05.0.1```
3. ```sbatch slurm/run.sbatch python libero/lifelong/pretrain.py```
