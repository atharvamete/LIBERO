seeds=(0 1 2 3 4 5 6 7 8 9)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python libero/lifelong/few_shot.py \
        pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip_old/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256/run_001/multitask_model_ep100.pth" \
        exp_name="rew_${seed}" \
        seed=$seed
done