seeds=(0 1 2 3 4 5 6 7 8 9)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python libero/lifelong/few_shot.py \
        pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip_old/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120/run_001/multitask_model_ep60.pth" \
        exp_name="rew_${seed}" \
        seed=$seed
done