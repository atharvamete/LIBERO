# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4_96_f4_k3s4_tt_n6d384/run_001/multitask_model_ep100.pth" \
#     exp_name="eval_m4_96_f4_k3s4_tt_n6d384" \
#     data.seq_len=96 \
#     policy.skill_block_size=96 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,2,1] \
#     policy.prior.block_size=24 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_90" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a1_l1h128_c32_n2_n6d120/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_lib90_vqbet_a1_l1h128_c32_n2_n6d120_ep80" \
    data.seq_len=10 \
    policy.skill_block_size=1 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=1 \
    benchmark_name="LIBERO_90" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/VQBet_Model/ResnetEncoder/vqbet_a5_l1h128_c32_n2_n6d120/run_001/multitask_model_ep60.pth" \
    exp_name="eval40_lib90_vqbet_a5_l1h128_c32_n2_n6d120_ep60" \
    data.seq_len=14 \
    policy.skill_block_size=5 \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    policy.codebook_size=32 \
    policy.gpt_n_embd=120 \
    policy.mpc_horizon=5 \
    benchmark_name="LIBERO_90" 

