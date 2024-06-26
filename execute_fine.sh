sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune" \
    tune_decoder=false \
    seed=0

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune_coef10_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune_coef10_offset2" \
    tune_decoder=false \
    policy.prior.offset_layers=2 \
    seed=0

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune_coef100_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune_coef100_offset2" \
    tune_decoder=false \
    policy.offset_loss_scale=100 \
    policy.prior.offset_layers=2 \
    seed=0

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef10/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef10" \
    seed=0

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef100/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef100" \
    policy.offset_loss_scale=100 \
    seed=0

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef10_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef10_offset2" \
    policy.offset_loss_scale=10 \
    policy.prior.offset_layers=2 \
    seed=0

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef100_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef100_offset2" \
    policy.offset_loss_scale=100 \
    policy.prior.offset_layers=2 \
    seed=0



sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune" \
    tune_decoder=false \
    seed=1

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune_coef10_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune_coef10_offset2" \
    tune_decoder=false \
    policy.prior.offset_layers=2 \
    seed=1

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune_coef100_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune_coef100_offset2" \
    tune_decoder=false \
    policy.offset_loss_scale=100 \
    policy.prior.offset_layers=2 \
    seed=1

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef10/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef10" \
    seed=1

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef100/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef100" \
    policy.offset_loss_scale=100 \
    seed=1

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef10_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef10_offset2" \
    policy.offset_loss_scale=10 \
    policy.prior.offset_layers=2 \
    seed=1

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef100_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef100_offset2" \
    policy.offset_loss_scale=100 \
    policy.prior.offset_layers=2 \
    seed=1




sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune" \
    tune_decoder=false \
    seed=2

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune_coef10_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune_coef10_offset2" \
    tune_decoder=false \
    policy.prior.offset_layers=2 \
    seed=2

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_notune_coef100_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_notune_coef100_offset2" \
    tune_decoder=false \
    policy.offset_loss_scale=100 \
    policy.prior.offset_layers=2 \
    seed=2

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef10/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef10" \
    seed=2

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef100/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef100" \
    policy.offset_loss_scale=100 \
    seed=2

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef10_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef10_offset2" \
    policy.offset_loss_scale=10 \
    policy.prior.offset_layers=2 \
    seed=2

sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py \
    pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/quest_coef100_offset2/run_002/multitask_model_ep100.pth' \
    exp_name="quest_coef100_offset2" \
    policy.offset_loss_scale=100 \
    policy.prior.offset_layers=2 \
    seed=2

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_notune" \
#     tune_decoder=false

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_notune_coef10_offset2" \
#     tune_decoder=false \
#     policy.prior.offset_layers=2

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_notune_coef100_offset2" \
#     tune_decoder=false \
#     policy.offset_loss_scale=100 \
#     policy.prior.offset_layers=2

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_coef10"

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_coef100" \
#     policy.offset_loss_scale=100

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_coef10_offset2" \
#     policy.offset_loss_scale=10 \
#     policy.prior.offset_layers=2

# sbatch slurm/eval.sbatch python libero/lifelong/few_shot.py \
#     pretrain_model_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f4_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth' \
#     exp_name="quest_coef100_offset2" \
#     policy.offset_loss_scale=100 \
#     policy.prior.offset_layers=2


