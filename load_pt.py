import torch

# Specify the path to the .pt file
file_path = "/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/evaluations_clip_old/LIBERO_90/final/vqbet/eval40_lib90_vqbet_a5_l1h128_c32_n2_n6d120_ep60/run_001/result.pt"

# Load the .pt file
data = torch.load(file_path)
sr = data['S_conf_mat'][-1][:8]
# Print the contents of the file
print(sr)