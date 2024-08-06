
# seeds=(0 1 2 3 4 5 6 7 8 9)
seeds=(1 2)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python libero/lifelong/skill_policy_eval.py \
        pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip_old/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth" \
        exp_name="rew_${seed}" \
        seed=$seed
done