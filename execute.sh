sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384/run_001/multitask_model_ep20.pth" \
    exp_name="eval20_lib_10_m4no_32_f4_k3s4_tt_n6d384" \
    seed=1 \
    benchmark_name="LIBERO_10"

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384/run_001/multitask_model_ep20.pth" \
    exp_name="eval20_lib_10_m4no_32_f4_k3s4_tt_n6d384" \
    seed=2 \
    benchmark_name="LIBERO_10"

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384/run_001/multitask_model_ep20.pth" \
    exp_name="eval20_lib_10_m4no_32_f4_k3s4_tt_n6d384" \
    seed=3 \
    benchmark_name="LIBERO_10"
