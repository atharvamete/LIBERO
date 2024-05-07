sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_single_stack_n6d384_off0_nobugs/run_001/multitask_model_ep100.pth" \
    exp_name="m4no_single_stack_n6d384_off0_5shot" \

sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_single_stack_n6d384_off10_nobugs/run_001/multitask_model_ep100.pth" \
    exp_name="m4no_single_stack_n6d384_off10_5shot" \
    policy.prior.offset_layers=2 \


