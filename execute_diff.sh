sbatch slurm/few.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120/run_001/multitask_model_ep60.pth" \
    exp_name="vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 
