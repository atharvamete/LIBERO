sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a1_l1h128_c16_n2/run_001/multitask_model_ep300.pth" \
    exp_name="vqbet_a1_l1h128_c16_n2_n6d120" \
    data.seq_len=10 \
    policy.skill_block_size=1 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=16 \
    policy.gpt_n_embd=120 

sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
    pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a5_l1h128_c16_n2/run_001/multitask_model_ep300.pth" \
    exp_name="vqbet_a5_l1h128_c16_n2_n6d120" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=16 \
    policy.gpt_n_embd=120 

# sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
#     pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a5_l2h512_c16_n2/run_001/multitask_model_ep300.pth" \
#     exp_name="vqbet_a5_l2h512_c16_n2_n6d120" \
#     data.seq_len=14 \
#     policy.skill_block_size=5 \
#     policy.hidden_dim=512 \
#     policy.num_layers=2 \
#     policy.codebook_size=16 \
#     policy.gpt_n_embd=120 

# sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py \
#     pretrain_vqvae_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_10/Multitask_Pretrain/VQVAE_Model/ResnetEncoder/vqbet_a5_l4h1024_c16_n2/run_001/multitask_model_ep300.pth" \
#     exp_name="vqbet_a5_l4h1024_c16_n2_n6d120" \
#     data.seq_len=14 \
#     policy.skill_block_size=5 \
#     policy.hidden_dim=1024 \
#     policy.num_layers=4 \
#     policy.codebook_size=16 \
#     policy.gpt_n_embd=120 
