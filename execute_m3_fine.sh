sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_full/run_001/multitask_model_ep50.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_full" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=120 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n6d384_5shot/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n6d384_5shot" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n6d384_full/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n6d384_full" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=6 \
    policy.prior.n_head=6 \
    policy.prior.n_embd=384 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n8d512_5shot/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n8d512_5shot" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n8d512_full/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n8d512_full" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n8d512_5shot_dec/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n8d512_5shot_dec" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n8d512_5shot_decobs/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n8d512_5shot_decobs" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 

sbatch slurm/run.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m3co_32_f4_k3s1_tt_n8d512_full_decobs/run_001/multitask_model_ep40.pth" \
    exp_name="eval_m3co_32_f4_k3s1_tt_n8d512_full_decobs" \
    policy.use_m4=0 \
    policy.cross_z=false \
    policy.use_causal_decoder=true \
    policy.strides=[1,1,1] \
    policy.prior.block_size=32 \
    policy.prior.n_layer=8 \
    policy.prior.n_head=8 \
    policy.prior.n_embd=512 