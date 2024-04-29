sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib90_diff_ddim100_10_l3d256" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_90" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib90_diff_ddim100_10_l5d256" \
    policy.down_dims=[256,256,512,512,1024] \
    benchmark_name="LIBERO_90" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_diff_ddim100_10_l3d256" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddpm100_l3d256/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_diff_ddpm100_100_l3d256" \
    policy.down_dims=[256,512,1024] \
    policy.scheduler="ddpm" \
    policy.diffusion_inf_steps=100 \
    benchmark_name="LIBERO_10" 


