sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep40.pth" \
    exp_name="eval40_m4_32_f4_k3s4_tt_n6d384_5shot_decobs_ep40" \
    data.seq_len=32 \
    policy.skill_block_size=32 \
    policy.kernel_sizes=[5,3,3] \
    policy.strides=[2,2,1] \
    policy.prior.block_size=8 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_m4_32_f4_k3s4_tt_n6d384_5shot_decobs_ep80" \
    data.seq_len=32 \
    policy.skill_block_size=32 \
    policy.kernel_sizes=[5,3,3] \
    policy.strides=[2,2,1] \
    policy.prior.block_size=8 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 \
    benchmark_name="LIBERO_10" 

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_m4_32_f4_k3s4_tt_n6d384_5shot_decobs_ep100" \
    data.seq_len=32 \
    policy.skill_block_size=32 \
    policy.kernel_sizes=[5,3,3] \
    policy.strides=[2,2,1] \
    policy.prior.block_size=8 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 \
    benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s1_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_m4_32_f4_k3s1_tt_n6d384_5shot_decobs_ep100" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[1,1,1] \
#     policy.prior.block_size=32 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s2_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_m4_32_f4_k3s2_tt_n6d384_5shot_decobs_ep100" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,1,1] \
#     policy.prior.block_size=16 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_m4_32_f4_k4s8_tt_n6d384_5shot_decobs_ep100" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3,3] \
#     policy.strides=[2,2,2,1] \
#     policy.prior.block_size=4 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_m4_64_f4_k3s4_tt_n6d384_5shot_decobs_ep100" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,2,1] \
#     policy.prior.block_size=16 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_m4_64_f4_k4s8_tt_n6d384_5shot_decobs_ep100" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3,3] \
#     policy.strides=[2,2,2,1] \
#     policy.prior.block_size=8 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k6s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep100.pth" \
#     exp_name="eval40_m4_64_f4_k6s8_tt_n6d384_5shot_decobs_ep100" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3,3,3,3] \
#     policy.strides=[2,1,2,1,2,1] \
#     policy.prior.block_size=8 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 


# #epoch 20
# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s1_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_m4_32_f4_k3s1_tt_n6d384_5shot_decobs_ep20" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[1,1,1] \
#     policy.prior.block_size=32 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s2_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_m4_32_f4_k3s2_tt_n6d384_5shot_decobs_ep20" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,1,1] \
#     policy.prior.block_size=16 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_m4_32_f4_k4s8_tt_n6d384_5shot_decobs_ep20" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3,3] \
#     policy.strides=[2,2,2,1] \
#     policy.prior.block_size=4 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_m4_64_f4_k3s4_tt_n6d384_5shot_decobs_ep20" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,2,1] \
#     policy.prior.block_size=16 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_m4_64_f4_k4s8_tt_n6d384_5shot_decobs_ep20" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3,3] \
#     policy.strides=[2,2,2,1] \
#     policy.prior.block_size=8 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k6s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth" \
#     exp_name="eval40_m4_64_f4_k6s8_tt_n6d384_5shot_decobs_ep20" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3,3,3,3] \
#     policy.strides=[2,1,2,1,2,1] \
#     policy.prior.block_size=8 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 


# #epoch 60
# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s1_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_m4_32_f4_k3s1_tt_n6d384_5shot_decobs_ep60" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[1,1,1] \
#     policy.prior.block_size=32 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s2_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_m4_32_f4_k3s2_tt_n6d384_5shot_decobs_ep60" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,1,1] \
#     policy.prior.block_size=16 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_m4_32_f4_k4s8_tt_n6d384_5shot_decobs_ep60" \
#     data.seq_len=32 \
#     policy.skill_block_size=32 \
#     policy.kernel_sizes=[5,3,3,3] \
#     policy.strides=[2,2,2,1] \
#     policy.prior.block_size=4 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_m4_64_f4_k3s4_tt_n6d384_5shot_decobs_ep60" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3] \
#     policy.strides=[2,2,1] \
#     policy.prior.block_size=16 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_m4_64_f4_k4s8_tt_n6d384_5shot_decobs_ep60" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3,3] \
#     policy.strides=[2,2,2,1] \
#     policy.prior.block_size=8 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 

# sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
#     pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k6s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth" \
#     exp_name="eval40_m4_64_f4_k6s8_tt_n6d384_5shot_decobs_ep60" \
#     data.seq_len=64 \
#     policy.skill_block_size=64 \
#     policy.kernel_sizes=[5,3,3,3,3,3] \
#     policy.strides=[2,1,2,1,2,1] \
#     policy.prior.block_size=8 \
#     policy.prior.n_layer=6 \
#     policy.prior.n_head=6 \
#     policy.prior.n_embd=384 \
#     benchmark_name="LIBERO_10" 
