import zipfile

def compress_files(file_paths, zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path)

# Example usage
file_paths = [
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s1_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s2_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k6s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s1_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k3s2_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_32_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k3s4_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k4s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_finetune_clip/LIBERO_10/Multitask/SkillGPT_Model/ResnetEncoder/m4_64_f4_k6s8_tt_n6d384_5shot_decobs/run_001/multitask_model_ep60.pth'
    ]
zip_file_name = 'ablation_fine_ep.zip'
compress_files(file_paths, zip_file_name)