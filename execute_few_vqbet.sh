sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec_ep100" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec_ep80" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec_ep50" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec_ep20" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_vqbet_a5_l1h128_c32_n2_n6d120_5shot_dec_ep10" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 
