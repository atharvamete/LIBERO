import os
import re

def extract_lines(filename):
    lines = []
    with open(filename, 'r') as file:
        # Read all lines into a list
        all_lines = file.readlines()
        # Extract line 57 and last 4th line
        line_57 = all_lines[56].strip() if len(all_lines) >= 57 else "Line 57 not found"
        line_minus_3 = all_lines[-3].strip() if len(all_lines) >= 3 else "Last 3rd line not found"
        lines.append(line_57)
        lines.append(line_minus_3)
    return lines

def main(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".out"):
            file_path = os.path.join(folder_path, filename)
            lines = extract_lines(file_path)
            print(lines[0])
            rates = re.findall(r'\d+\.\d+', lines[1])
            rates = [float(rate) for rate in rates]
            sr = sum(rates) / len(rates)
            print(f"Success rate: {sr}")
            print()

if __name__ == "__main__":
    folder_path = "/storage/home/hcoda1/0/amete7/p-agarg35-0/diff-skill/LIBERO/slurm_out_diff_few"
    main(folder_path)
