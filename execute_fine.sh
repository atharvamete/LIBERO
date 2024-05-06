sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth" \
    exp_name="m4no_32_f4_k3s4_tt_n6d384_off0_5shot" \

sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off10/run_001/multitask_model_ep100.pth" \
    exp_name="m4no_32_f4_k3s4_tt_n6d384_off10_5shot" \
    policy.prior.offset_layers=2 \


 