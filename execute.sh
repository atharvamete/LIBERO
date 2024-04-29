sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_diff_ddim100_l3d256_5shot_ep100" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_diff_ddim100_l3d256_5shot_ep80" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_diff_ddim100_l3d256_5shot_ep50" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_diff_ddim100_l3d256_5shot_ep20" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_diff_ddim100_l3d256_5shot_ep10" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

# l5d256
sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_diff_ddim100_l5d256_5shot_ep100" \
    policy.down_dims=[256,256,512,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_diff_ddim100_l5d256_5shot_ep80" \
    policy.down_dims=[256,256,512,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_diff_ddim100_l5d256_5shot_ep50" \
    policy.down_dims=[256,256,512,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_diff_ddim100_l5d256_5shot_ep20" \
    policy.down_dims=[256,256,512,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_diff_ddim100_l5d256_5shot_ep10" \
    policy.down_dims=[256,256,512,512,1024] \
    benchmark_name="LIBERO_10" 
