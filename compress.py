import zipfile

def compress_files(file_paths, zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path)

# Example usage
file_paths = [
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256/run_001/multitask_model_ep100.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_90/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l4d256/run_001/multitask_model_ep100.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddim100_l3d256/run_001/multitask_model_ep100.pth',
    '/satassdscratch/amete7/LIBERO/experiments_clip/LIBERO_10/Multitask/Diffusion_Policy/ResnetEncoder/diff_ddpm100_l3d256/run_001/multitask_model_ep100.pth'
    ]
zip_file_name = 'diffusion_pol.zip'
compress_files(file_paths, zip_file_name)