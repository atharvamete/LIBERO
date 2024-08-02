seeds=(0 1 2 3 4 5 6 7 8 9)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python libero/lifelong/skill_policy_eval.py \
        pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/rew_${seed}/run_001/multitask_model_ep180.pth" \
        exp_name="rew_${seed}" \
        seed=$seed
done