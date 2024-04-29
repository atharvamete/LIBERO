sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256/run_001/multitask_model_ep100.pth" \
    exp_name="diff_ddim100_l3d256_5shot" \
    policy.down_dims=[256,512,1024] \
    benchmark_name="LIBERO_10" 

# sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l5d256/run_001/multitask_model_ep100.pth" \
#     exp_name="diff_ddim100_l5d256_5shot" \
#     policy.down_dims=[256,256,512,512,1024] \
#     benchmark_name="LIBERO_10" 



