import os
import re
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--clean_ckpt_pat", help="pattern to match checkpoint folders")
parser.add_argument("--skip_ckpt_pat", default=None, help="pattern to skip checkpoint folders")
parser.add_argument("--del_samples", help="delete samples", action="store_true")
args = parser.parse_args()

def delete_all_but_largest_checkpoint(ckpt_folder_path):
    # Regex to match checkpoint files like embeddings_gs-5500.pt and extract the iteration number
    checkpoint_pattern = re.compile(r'embeddings_gs-(\d+)\.pt')
    
    checkpoints = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(ckpt_folder_path):
        match = checkpoint_pattern.match(filename)
        if match:
            iteration = int(match.group(1))  # Extract the iteration number
            checkpoints.append((iteration, filename))
    
    if not checkpoints:
        print(f"No checkpoints found in {ckpt_folder_path}")
        return
    
    # Find the checkpoint with the largest iteration number
    largest_iteration, largest_checkpoint = max(checkpoints, key=lambda x: x[0])
    
    # Delete all other checkpoints
    for iteration, filename in checkpoints:
        if iteration != largest_iteration:
            file_path = os.path.join(ckpt_folder_path, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    
    print(f"Kept:    {os.path.join(ckpt_folder_path, largest_checkpoint)}")

def clean_log_folders(root_log_folder, clean_ckpt_pat, skip_ckpt_pat, del_samples):
    # Iterate through all subfolders (log folders)
    for subfolder in os.listdir(root_log_folder):
        ckpt_folder_path = os.path.join(root_log_folder, subfolder, 'checkpoints')
        if os.path.isdir(ckpt_folder_path) and re.search(clean_ckpt_pat, ckpt_folder_path):
            if skip_ckpt_pat and re.search(skip_ckpt_pat, ckpt_folder_path):
                print(f"Skipping: {ckpt_folder_path}")
                continue
            print(f"Cleaning: {ckpt_folder_path}")
            delete_all_but_largest_checkpoint(ckpt_folder_path)
        else:
            print(f"Skipping: {ckpt_folder_path}")

        if del_samples:
            samples_folder_path = os.path.join(root_log_folder, subfolder, 'samples')
            print(f"Deleting: {samples_folder_path}")
            os.system(f"rm -rf {samples_folder_path}")
            
if __name__ == "__main__":
    # Set the path to the root folder containing all log folders
    root_log_folder = "logs/"  # Update this path
    clean_log_folders(root_log_folder, clean_ckpt_pat=args.clean_ckpt_pat, skip_ckpt_pat=args.skip_ckpt_pat, 
                      del_samples=args.del_samples)
