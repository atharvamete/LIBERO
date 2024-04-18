## Running this on PACE

1. Start a free interactive session with a RTX6000 GPU: ```salloc -A gts-agarg35 -N1 --mem-per-gpu=12G -q embers -t8:00:00 --gres=gpu:RTX_6000:1```
2. Load in Conda: ```module load anaconda3/2022.05.0.1```

Command to train stage 1:  
1. ```tmux new -s train1```  
2. ```export CUDA_VISIBLE_DEVICES=7 && export MUJOCO_EGL_DEVICE_ID=7 && python libero/lifelong/pretrain.py```
3. ```export CUDA_VISIBLE_DEVICES=7 && export MUJOCO_EGL_DEVICE_ID=7 && python libero/lifelong/skill_policy.py```
4. ```export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python libero/lifelong/skill_policy_eval.py```
5. ```export CUDA_VISIBLE_DEVICES=7 && export MUJOCO_EGL_DEVICE_ID=7 && python libero/lifelong/few_shot.py```
