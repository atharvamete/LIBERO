# Self-Supervised Skill Abstraction and Decision Making

## Model specific installations
0. ```conda with python 3.8```
1. ```pip install vector-quantize-pytorch==1.8.1```
2. ```pip install positional-encodings==6.0.3```
3. ```pip install einops==0.3.2```
4. ```pip install hydra-core==1.2.0```
5. Other dependencies like torch, torchvision, transformers, etc
6. Libero specific dependencies are in the `requirements.txt` file.

## Model Files
Model files are in the `libero/lifelong/models` directory. Description:
1. `skill_vae.py`: stage 1 policy file, uses base_policy class, imports stage 1 model and observation encoder.
2. `modules/skill_vae_modules.py`: SkillVAE model - stage 1.
3. `skill_GPT.py`: stage 2 policy file, uses base_policy class, imports stage 1 model and stage 2 GPT model both.
4. `modules/skill_utils.py`: contains stage 2 GPT model and other stage 1 modules.
5. `modules/rgb_modules.py`: contains visual observation encoder.

## Config Files
Config files are in the `libero/configs` directory. Description:
1. `pretrain.yaml`: stage 1 training.
2. `skillGPT.yaml`: stage 2 training.
3. `few_shot.yaml`: few-shot training on libero-10.
All of these files use default files in the `libero/configs` directory.  
Following are model specific config files:
1. `policy/skill_vae.yaml`: stage 1 model.
2. `policy/skill_GPT.yaml`: stage 2 model.

Read comments in the config files for more details.  

## Training Scripts
These are libero specific scripts that loads the model and data and environment (for evals) and trains the model.  
Main scripts for training are in the `libero/lifelong` directory. The main scripts are:
1. `pretrain.py`: Pretrain the SkillVAE model.
2. `skill_policy.py`: Train the skill policy.
These scripts load `algo/multitask.py` algo that loads multitask data and trains the model. (libero specific)

1. Command to train stage 1:  
```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/pretrain.py```
2. Command to train stage 2:
```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/skill_policy.py```
3. Command to finetune:
```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/few_shot.py```
4. Command to evaluate:
```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/skill_policy_eval.py```

## Running this on PACE

1. Start a free interactive session with a RTX6000 GPU: ```salloc -A gts-agarg35 -N1 --mem-per-gpu=24G -q embers -t8:00:00 --gres=gpu:RTX_6000:1```
2. Load in Conda: ```module load anaconda3/2022.05.0.1```
3. ```sbatch slurm/run.sbatch python libero/lifelong/pretrain.py```

## PACE commands
1. ```squeue -u <username>``` to check the status of the job.
2. ```scancel <job_id>``` to cancel the job.
3. ```pace-check-queue -c gpu-a100``` to check the queue for A100 GPUs.
4. ```pace=quota``` to check the quota.
5. ```sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py``` to use inferno-paid A100-50hrs for training. (A100 is faster than RTX6000)
6. ```sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py``` to use embers-free RTX6000-8hrs for eval. (RTX6000 is suitable for rendering)
7. ```salloc -A gts-agarg35 -N1 --mem-per-gpu=32G -q embers -t8:00:00 --gres=gpu:V100:1``` to start a job on embers-free RTX6000-8hrs.
8. ```salloc -A gts-agarg35 -N1 --mem-per-gpu=32G -q inferno -t50:00:00 --gpus=V100:1 --constraint V100-32GB``` to start a job on inferno-paid V100-32GB-50hrs.