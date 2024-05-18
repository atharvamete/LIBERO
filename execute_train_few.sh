sbatch slurm/few.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l1h128_c32_n2_n6d120/run_016/multitask_model.pth" \
    exp_name="vqbet_a32_l1h128_c32_n2_n6d120_5shot_dec" \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 

sbatch slurm/few.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l2h256_c32_n2_n6d120/run_015/multitask_model.pth" \
    exp_name="vqbet_a32_l2h256_c32_n2_n6d120_5shot_dec" \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=256 \
    policy.num_layers=2 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 
