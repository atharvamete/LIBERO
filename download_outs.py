import os
import shutil

# Get the current directory
current_dir = os.getcwd()

# Get a list of all folders in the current directory
folders = [folder for folder in os.listdir(current_dir) if os.path.isdir(folder)]

# Compress each folder that starts with "slurm_"
for folder in folders:
    if folder.startswith("slurm_"):
        shutil.make_archive('all_slurm_folders', 'zip', current_dir, folder)