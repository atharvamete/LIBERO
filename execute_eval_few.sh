sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off0_5shot_ep10" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off10_5shot/run_001/multitask_model_ep10.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off10_5shot_ep10" \
    policy.prior.offset_layers=2 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off0_5shot_ep20" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off10_5shot/run_001/multitask_model_ep20.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off10_5shot_ep20" \
    policy.prior.offset_layers=2 \
    benchmark_name="LIBERO_10" \


sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off0_5shot_ep50" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off10_5shot/run_001/multitask_model_ep50.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off10_5shot_ep50" \
    policy.prior.offset_layers=2 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off0_5shot_ep80" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off10_5shot/run_001/multitask_model_ep80.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off10_5shot_ep80" \
    policy.prior.offset_layers=2 \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off0_5shot_ep100" \
    benchmark_name="LIBERO_10" \

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off10_5shot/run_001/multitask_model_ep100.pth" \
    exp_name="eval40_lib10_m4no_32_f4_k3s4_tt_n6d384_off10_5shot_ep100" \
    policy.prior.offset_layers=2 \
    benchmark_name="LIBERO_10" \

