import os
import torch

def open_results(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'result.pt':
                file_path = os.path.join(root, file)
                data = torch.load(file_path)
                sr = data['S_conf_mat'][-1]
                # Print the contents of the file
                print(sr)

# Replace '/path/to/directory' with the actual directory path
directory_path = '/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/evaluations_clip_old/LIBERO_90/final/diffusion/eval20_lib90_diff_ddim100_l3d256'
open_results(directory_path)