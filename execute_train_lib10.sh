sbatch slurm/train.sbatch python libero/lifelong/pretrain.py \
    exp_name="vqbet_a32_l1h128_c32_n2_n6d120" \
    policy.hidden_dim=128 \
    policy.num_layers=1 \
    benchmark_name="LIBERO_10" 

sbatch slurm/train.sbatch python libero/lifelong/pretrain.py \
    exp_name="vqbet_a32_l2h256_c32_n2_n6d120" \
    policy.hidden_dim=256 \
    policy.num_layers=2 \
    benchmark_name="LIBERO_10" 

