import os
import re

def find_largest_checkpoint(folder_path):
    # Regex to match checkpoint files like embeddings_gs-5500.pt and extract the iteration number
    checkpoint_pattern = re.compile(r'embeddings_gs-(\d+)\.pt')
    
    checkpoints = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        match = checkpoint_pattern.match(filename)
        if match:
            iteration = int(match.group(1))  # Extract the iteration number
            checkpoints.append((iteration, filename))
    
    if not checkpoints:
        print(f"No checkpoints found in {folder_path}")
        return
    
    # Find the checkpoint with the largest iteration number
    largest_iteration, largest_checkpoint = max(checkpoints, key=lambda x: x[0])
    
    # Delete all other checkpoints
    for iteration, filename in checkpoints:
        if iteration != largest_iteration:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    
    print(f"Kept:    {os.path.join(folder_path, largest_checkpoint)}")

def clean_log_folders(root_log_folder):
    # Iterate through all subfolders (log folders)
    for subfolder in os.listdir(root_log_folder):
        folder_path = os.path.join(root_log_folder, subfolder, 'checkpoints')
        if os.path.isdir(folder_path):
            find_largest_checkpoint(folder_path)

if __name__ == "__main__":
    # Set the path to the root folder containing all log folders
    root_log_folder = "logs/"  # Update this path
    clean_log_folders(root_log_folder)
