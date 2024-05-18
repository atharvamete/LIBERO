sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a32_l2h128_c32_n2/run_001/multitask_model_ep100.pth" \
    exp_name="vqbet_a32_l1h128_c32_n2_n6d120" \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.gpt_n_embd=120 \
    policy.gpt_block_size=50 \
    benchmark_name="LIBERO_90" \
    continue_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l1h128_c32_n2_n6d120/run_016/multitask_model.pth"

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a32_l2h128_c32_n2/run_001/multitask_model_ep100.pth" \
    exp_name="vqbet_a32_l1h128_c32_n2_n6d120_nooff" \
    policy.offset_loss_multiplier=0.0 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.gpt_n_embd=120 \
    policy.gpt_block_size=50 \
    benchmark_name="LIBERO_90" \
    continue_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l1h128_c32_n2_n6d120_nooff/run_011/multitask_model.pth"

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a32_l2h256_c32_n2/run_001/multitask_model_ep100.pth" \
    exp_name="vqbet_a32_l2h256_c32_n2_n6d120" \
    policy.hidden_dim=256 \
    policy.num_layers=2 \
    policy.gpt_n_embd=120 \
    policy.gpt_block_size=50 \
    benchmark_name="LIBERO_90" \
    continue_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l2h256_c32_n2_n6d120/run_015/multitask_model.pth"

