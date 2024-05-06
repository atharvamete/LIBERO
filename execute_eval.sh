sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib90_m4op_32_f4_k3s4_tt_n6d384_off0" \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4op_8_f4_k3s4_tt_n6d384_off100/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_lib90_m4op_8_f4_k3s4_tt_n6d384_off100" \
#     data.seq_len=8 \
#     policy.skill_block_size=8 \
#     policy.prior.block_size=2 \
#     policy.offset_loss_scale=100 \

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f5_k3s4_tt_n6d384/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_lib90_m4op_32_f5_k3s4_tt_n6d384" \
#     policy.fsq_level=[7,5,5,5,5] \
#     policy.offset_loss_scale=1 \
#     policy.prior.vocab_size=4380 \
#     policy.prior.output_dim=4375 \
#     policy.prior.start_token=4376 \



