sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_single_stack_n6d384_off0_nobugs/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib90_m4no_single_stack_n6d384_off0" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_single_stack_n6d384_off10_nobugs/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib90_m4no_single_stack_n6d384_off10" \
    policy.prior.offset_layers=2 \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_single_stack_n6d384_off0_nobugs/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4no_single_stack_n6d384_off0_zeroshot" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_single_stack_n6d384_off10_nobugs/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4no_single_stack_n6d384_off10_zeroshot" \
    policy.prior.offset_layers=2 \
    benchmark_name="LIBERO_10" \

