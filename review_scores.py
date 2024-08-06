import os
import json
import numpy as np
import torch

directories = [
    "/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/evaluations_clip/LIBERO_10/Multitask/SkillGPT_Model",
    "/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/evaluations_clip/LIBERO_10/Multitask/Diffusion_Policy",
    "/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/evaluations_clip/LIBERO_10/Multitask/VQBet_Model",
]

include = None
exclude = None
scores = {}

for directory in directories:
    if "SkillGPT_Model" in directory:
        include = "ep180"
    else:
        include = None
    success_rates = []
    for root, dirs, files in os.walk(directory):
        if 'result.pt' in files:
            if include is not None and include not in root or exclude is not None and exclude in root:
                continue
            result = torch.load(os.path.join(root, 'result.pt'))
            rollout_success_rate = result['S_conf_mat'][-1][:8]
            success_rates.append(rollout_success_rate)
    name = directory.split('/')[-1]
    success_rates = np.array(success_rates)
    print(f'{name}: mean: {success_rates.mean()}, std: {success_rates.mean(axis=1).std()}')
    scores[name] = success_rates

# save scores to file
np.savez('rest_few.npz', **scores)