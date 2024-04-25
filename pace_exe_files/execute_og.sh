sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n6d384/multitask_model_ep100.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n6d384" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n8d512/multitask_model_ep100.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n8d512" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s4_tt_n6d384/multitask_model_ep100.pth" \
    exp_name="eval_m4_32_f4_k3s4_tt_n6d384" \
    policy.use_m4=1 \
    policy.cross_z=true \
    policy.use_causal_decoder=true \
    policy.strides=[2,2,1] \
    policy.prior.block_size=8 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s4_tt_n8d512/multitask_model_ep100.pth" \
    exp_name="eval_m4_32_f4_k3s4_tt_n8d512" \
    policy.use_m4=1 \
    policy.cross_z=true \
    policy.use_causal_decoder=true \
    policy.strides=[2,2,1] \
    policy.prior.block_size=8 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s4_tt_n8d768/multitask_model_ep100.pth" \
    exp_name="eval_m4_32_f4_k3s4_tt_n8d768" \
    policy.use_m4=1 \
    policy.cross_z=true \
    policy.use_causal_decoder=true \
    policy.strides=[2,2,1] \
    policy.prior.block_size=8 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=768 