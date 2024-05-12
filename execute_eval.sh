sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f16_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib90_m4no_32_f16_k3s4_tt_n6d384_off0_ep20" \
    policy.skill_vae_1.fsq_level=[5,3] \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f64_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib90_m4no_32_f64_k3s4_tt_n6d384_off0_ep20" \
    policy.skill_vae_1.fsq_level=[8,8] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f256_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_lib90_m4no_32_f256_k3s4_tt_n6d384_off0_ep20" \
#     policy.skill_vae_1.fsq_level=[8,6,5] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f512_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_lib90_m4no_32_f512_k3s4_tt_n6d384_off0_ep20" \
#     policy.skill_vae_1.fsq_level=[8,8,8] \


# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f16_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_lib90_m4no_32_f16_k3s4_tt_n6d384_off0_ep60" \
#     policy.skill_vae_1.fsq_level=[5,3] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f64_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_lib90_m4no_32_f64_k3s4_tt_n6d384_off0_ep60" \
#     policy.skill_vae_1.fsq_level=[8,8] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f256_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_lib90_m4no_32_f256_k3s4_tt_n6d384_off0_ep60" \
#     policy.skill_vae_1.fsq_level=[8,6,5] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f512_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_lib90_m4no_32_f512_k3s4_tt_n6d384_off0_ep60" \
#     policy.skill_vae_1.fsq_level=[8,8,8] \


# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f16_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_lib90_m4no_32_f16_k3s4_tt_n6d384_off0_ep100" \
#     policy.skill_vae_1.fsq_level=[5,3] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f64_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_lib90_m4no_32_f64_k3s4_tt_n6d384_off0_ep100" \
#     policy.skill_vae_1.fsq_level=[8,8] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f256_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_lib90_m4no_32_f256_k3s4_tt_n6d384_off0_ep100" \
#     policy.skill_vae_1.fsq_level=[8,6,5] \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f512_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_lib90_m4no_32_f512_k3s4_tt_n6d384_off0_ep100" \
#     policy.skill_vae_1.fsq_level=[8,8,8] \

