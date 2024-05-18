sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c16_n2_n6d120/run_001/multitask_model_ep160.pth" \
    exp_name="eval20_lib10_vqbet_a5_l1h128_c16_n2_n6d120_ep60" \
    seed=1 \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=16 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c16_n2_n6d120/run_001/multitask_model_ep160.pth" \
    exp_name="eval20_lib10_vqbet_a5_l1h128_c16_n2_n6d120_ep60" \
    seed=2 \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=16 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c16_n2_n6d120/run_001/multitask_model_ep160.pth" \
    exp_name="eval20_lib10_vqbet_a5_l1h128_c16_n2_n6d120_ep60" \
    seed=3 \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=16 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l1h128_c32_n2_n6d120/run_007/multitask_model_ep98.pth" \
    exp_name="eval20_lib10_vqbet_a32_l1h128_c32_n2_n6d120_ep98" \
    seed=1 \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l1h128_c32_n2_n6d120/run_007/multitask_model_ep98.pth" \
    exp_name="eval20_lib10_vqbet_a32_l1h128_c32_n2_n6d120_ep98" \
    seed=2 \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l1h128_c32_n2_n6d120/run_007/multitask_model_ep98.pth" \
    exp_name="eval20_lib10_vqbet_a32_l1h128_c32_n2_n6d120_ep98" \
    seed=3 \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l2h256_c32_n2_n6d120/run_007/multitask_model_ep104.pth" \
    exp_name="eval20_lib10_vqbet_a32_l2h256_c32_n2_n6d120_ep30" \
    seed=1 \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=256 \
    policy.num_layers=2 \
    policy.codebook_size=32 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l2h256_c32_n2_n6d120/run_007/multitask_model_ep104.pth" \
    exp_name="eval20_lib10_vqbet_a32_l2h256_c32_n2_n6d120_ep30" \
    seed=2 \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=256 \
    policy.num_layers=2 \
    policy.codebook_size=32 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask/VQBet_Model/ResnetEncoder/vqbet_a32_l2h256_c32_n2_n6d120/run_007/multitask_model_ep104.pth" \
    exp_name="eval20_lib10_vqbet_a32_l2h256_c32_n2_n6d120_ep30" \
    seed=3 \
    data.seq_len=41 \
    policy.skill_block_size=32 \
    policy.hidden_dim=256 \
    policy.num_layers=2 \
    policy.codebook_size=32 \
    policy.gpt_block_size=50 \
    policy.mpc_horizon=16 \
    benchmark_name="LIBERO_10" 


