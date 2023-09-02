import os
import shutil
import random

# Define the main directory
main_dir = 'data/data'

# Create train and val directories if they don't exist
train_dir = 'data/train'
val_dir = 'data/val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Go through each subdirectory in the main directory
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    
    if os.path.isdir(subdir_path):
        # Create corresponding subdirectories in train and val
        train_subdir = os.path.join(train_dir, subdir)
        val_subdir = os.path.join(val_dir, subdir)

        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(val_subdir, exist_ok=True)

        # List all image files in the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

        # Shuffle image files and split into train and val sets
        random.shuffle(image_files)
        train_files = image_files[:int(len(image_files) * 0.8)]
        val_files = image_files[int(len(image_files) * 0.8):]

        # Copy image files to the corresponding train and val subdirectories
        for train_file in train_files:
            shutil.copy2(os.path.join(subdir_path, train_file), os.path.join(train_subdir, train_file))

        for val_file in val_files:
            shutil.copy2(os.path.join(subdir_path, val_file), os.path.join(val_subdir, val_file))

print("Data split into train and val directories successfully.")
