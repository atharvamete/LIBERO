# sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
#     pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a1_l1h128_c32_n2/run_001/multitask_model_ep300.pth" \
#     exp_name="vqbet_a1_l1h128_c32_n2_n6d120" \
#     data.seq_len=10 \
#     policy.skill_block_size=1 \
#     policy.hidden_dim=128 \
#     policy.num_layers=1 \
#     policy.codebook_size=32 \
#     policy.gpt_n_embd=120 \
#     benchmark_name="LIBERO_90" 

# sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
#     pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a1_l4h1024_c32_n2/run_001/multitask_model_ep300.pth" \
#     exp_name="vqbet_a1_l4h1024_c32_n2_n6d120" \
#     data.seq_len=10 \
#     policy.skill_block_size=1 \
#     policy.hidden_dim=1024 \
#     policy.num_layers=4 \
#     policy.codebook_size=32 \
#     policy.gpt_n_embd=120 \
#     benchmark_name="LIBERO_90" 

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2/run_001/multitask_model_ep300.pth" \
    exp_name="vqbet_a5_l1h128_c32_n2_n6d120" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    benchmark_name="LIBERO_90" 

# sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
#     pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a5_l4h1024_c32_n2/run_001/multitask_model_ep300.pth" \
#     exp_name="vqbet_a5_l4h1024_c32_n2_n6d120" \
#     data.seq_len=14 \
#     policy.skill_block_size=5 \
#     policy.hidden_dim=1024 \
#     policy.num_layers=4 \
#     policy.codebook_size=32 \
#     policy.gpt_n_embd=120 \
#     benchmark_name="LIBERO_90" 

# sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
#     pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a32_l4h1024_c32_n2/run_001/multitask_model_ep300.pth" \
#     exp_name="vqbet_a32_l4h1024_c32_n2_n6d120" \
#     data.seq_len=41 \
#     policy.skill_block_size=32 \
#     policy.hidden_dim=1024 \
#     policy.num_layers=4 \
#     policy.codebook_size=32 \
#     policy.gpt_n_embd=120 \
#     policy.gpt_block_size=50 \
#     benchmark_name="LIBERO_90" 
