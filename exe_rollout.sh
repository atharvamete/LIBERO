sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth" \
    exp_name="video_m4no_multiIL" \
    benchmark_name="LIBERO_90" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384/run_001/multitask_model_ep20.pth" \
    exp_name="video_m4no_longIL" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="video_m4no_fewIL" \
    benchmark_name="LIBERO_10" \


