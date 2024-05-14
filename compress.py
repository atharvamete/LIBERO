import zipfile

def compress_files(file_paths, zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path)

# Example usage
file_paths = [
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f16_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f16_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f16_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f64_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f64_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f64_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f256_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f256_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f256_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f512_k3s4_tt_n6d384_off0/run_001/multitask_model_ep20.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f512_k3s4_tt_n6d384_off0/run_001/multitask_model_ep60.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/SkillGPT_Model/ResnetEncoder/m4no_32_f512_k3s4_tt_n6d384_off0/run_001/multitask_model_ep100.pth',
    ]
zip_file_name = 'codebook_size.zip'
compress_files(file_paths, zip_file_name)