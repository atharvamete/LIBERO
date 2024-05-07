sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/codec_rfq_n6d384_off0/run_001/multitask_model_ep90.pth" \
    exp_name="codec_rfq_n6d384_off0_5shot" \

sbatch slurm/train.sbatch python libero/lifelong/few_shot.py \
    pretrain_model_path="/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/codec_rvq_n6d384_off0/run_001/multitask_model_ep90.pth" \
    exp_name="codec_rvq_n6d384_off0_5shot" \
    policy.prior.start_token=32 \
    policy.prior.vocab_size=[8,8,16] \
    policy.codebook_dim=512 \
    policy.quantizer_args.quantizer_type="residual_vq" \
    policy.quantizer_args.dim=512 \
    policy.quantizer_args.codebook_size=[8,8,16] \
    policy.quantizer_args.quantizer_type="residual_vq" \


