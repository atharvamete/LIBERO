sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_lib10_m4op_32_f4_k3s4_tt_n6d384_off0_5shot_ep10" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_8_f4_k3s4_tt_n6d384_off100_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_lib10_m4op_8_f4_k3s4_tt_n6d384_off100_5shot_ep10" \
    data.seq_len=8 \
    policy.skill_block_size=8 \
    policy.prior.block_size=2 \
    policy.offset_loss_scale=100 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f5_k3s4_tt_n6d384_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_lib10_m4op_32_f5_k3s4_tt_n6d384_5shot_ep10" \
    policy.fsq_level=[7,5,5,5,5] \
    policy.offset_loss_scale=1 \
    policy.prior.vocab_size=4380 \
    policy.prior.output_dim=4375 \
    policy.prior.start_token=4376 \
    benchmark_name="LIBERO_10" \


sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib10_m4op_32_f4_k3s4_tt_n6d384_off0_5shot_ep20" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_8_f4_k3s4_tt_n6d384_off100_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib10_m4op_8_f4_k3s4_tt_n6d384_off100_5shot_ep20" \
    data.seq_len=8 \
    policy.skill_block_size=8 \
    policy.prior.block_size=2 \
    policy.offset_loss_scale=100 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f5_k3s4_tt_n6d384_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib10_m4op_32_f5_k3s4_tt_n6d384_5shot_ep20" \
    policy.fsq_level=[7,5,5,5,5] \
    policy.offset_loss_scale=1 \
    policy.prior.vocab_size=4380 \
    policy.prior.output_dim=4375 \
    policy.prior.start_token=4376 \
    benchmark_name="LIBERO_10" \



sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_lib10_m4op_32_f4_k3s4_tt_n6d384_off0_5shot_ep50" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_8_f4_k3s4_tt_n6d384_off100_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_lib10_m4op_8_f4_k3s4_tt_n6d384_off100_5shot_ep50" \
    data.seq_len=8 \
    policy.skill_block_size=8 \
    policy.prior.block_size=2 \
    policy.offset_loss_scale=100 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f5_k3s4_tt_n6d384_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_lib10_m4op_32_f5_k3s4_tt_n6d384_5shot_ep50" \
    policy.fsq_level=[7,5,5,5,5] \
    policy.offset_loss_scale=1 \
    policy.prior.vocab_size=4380 \
    policy.prior.output_dim=4375 \
    policy.prior.start_token=4376 \
    benchmark_name="LIBERO_10" \


sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_lib10_m4op_32_f4_k3s4_tt_n6d384_off0_5shot_ep80" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_8_f4_k3s4_tt_n6d384_off100_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_lib10_m4op_8_f4_k3s4_tt_n6d384_off100_5shot_ep80" \
    data.seq_len=8 \
    policy.skill_block_size=8 \
    policy.prior.block_size=2 \
    policy.offset_loss_scale=100 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f5_k3s4_tt_n6d384_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_lib10_m4op_32_f5_k3s4_tt_n6d384_5shot_ep80" \
    policy.fsq_level=[7,5,5,5,5] \
    policy.offset_loss_scale=1 \
    policy.prior.vocab_size=4380 \
    policy.prior.output_dim=4375 \
    policy.prior.start_token=4376 \
    benchmark_name="LIBERO_10" \



sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4op_32_f4_k3s4_tt_n6d384_off0_5shot_ep100" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_8_f4_k3s4_tt_n6d384_off100_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4op_8_f4_k3s4_tt_n6d384_off100_5shot_ep100" \
    data.seq_len=8 \
    policy.skill_block_size=8 \
    policy.prior.block_size=2 \
    policy.offset_loss_scale=100 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4op_32_f5_k3s4_tt_n6d384_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4op_32_f5_k3s4_tt_n6d384_5shot_ep100" \
    policy.fsq_level=[7,5,5,5,5] \
    policy.offset_loss_scale=1 \
    policy.prior.vocab_size=4380 \
    policy.prior.output_dim=4375 \
    policy.prior.start_token=4376 \
    benchmark_name="LIBERO_10" \



