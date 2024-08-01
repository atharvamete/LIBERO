sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_f4_k3s1_tt/run_001/multitask_model_ep100.pth' \
    exp_name="m4no_32_f4_k3s1_tt_n6d384" \
    policy.prior.block_size=32 \
    policy.skill_vae_1.kernel_sizes=[5,3,3] \
    policy.skill_vae_1.strides=[1,1,1]

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_f4_k4s8_tt/run_001/multitask_model_ep100.pth' \
    exp_name="m4no_32_f4_k4s8_tt_n6d384" \
    policy.prior.block_size=4 \
    policy.skill_vae_1.kernel_sizes=[5,3,3,3] \
    policy.skill_vae_1.strides=[2,2,2,1]

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_f4_k5s16_tt/run_001/multitask_model_ep100.pth' \
    exp_name="m4no_32_f4_k5s16_tt_n6d384" \
    policy.prior.block_size=2 \
    policy.skill_vae_1.kernel_sizes=[5,3,3,3,3] \
    policy.skill_vae_1.strides=[2,2,2,2,1]

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_64_f4_k4s8_tt/run_001/multitask_model_ep100.pth' \
    exp_name="m4no_64_f4_k4s8_tt_n6d384" \
    data.seq_len=64 \
    policy.skill_vae_1.skill_block_size=64 \
    policy.prior.block_size=8 \
    policy.skill_vae_1.kernel_sizes=[5,3,3,3] \
    policy.skill_vae_1.strides=[2,2,2,1]

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_64_f4_k5s16_tt/run_001/multitask_model_ep100.pth' \
    exp_name="m4no_64_f4_k5s16_tt_n6d384" \
    data.seq_len=64 \
    policy.skill_vae_1.skill_block_size=64 \
    policy.prior.block_size=4 \
    policy.skill_vae_1.kernel_sizes=[5,3,3,3,3] \
    policy.skill_vae_1.strides=[2,2,2,2,1]

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    exp_name="m4no_32_f4_k3s4_tf_n6d384" \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_f4_k3s4_tf/run_001/multitask_model_ep60.pth' \
    policy.skill_vae_1.use_causal_encoder=true \
    policy.skill_vae_1.use_causal_decoder=false \

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    exp_name="m4no_32_f4_k3s4_ft_n6d384" \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_f4_k3s4_ft/run_001/multitask_model_ep80.pth' \
    policy.skill_vae_1.use_causal_encoder=false \
    policy.skill_vae_1.use_causal_decoder=true \

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    exp_name="m4no_32_f4_k3s4_ff_n6d384" \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_f4_k3s4_ff/run_001/multitask_model_ep60.pth' \
    policy.skill_vae_1.use_causal_encoder=false \
    policy.skill_vae_1.use_causal_decoder=false \

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    exp_name="m4no_32_vq_k3s4_tt_n6d384" \
    policy.skill_vae_1.path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/SkillVAE_Model/ResnetEncoder/m4no_32_vq_k3s4_tt/run_001/multitask_model_ep60.pth'
    policy.skill_vae_1.vq_type="vq" \
    policy.prior.start_token=1024 \
    policy.prior.vocab_size=1024