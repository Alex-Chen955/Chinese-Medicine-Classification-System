import os
def generate_label(dir_path, output_file):
    # Get a list of all subdirectories in the directory
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # Open the output file for writing
    with open(output_file, "w") as f:
        # Iterate over the subdirectories and their files
        for i, subdir in enumerate(subdirs):
            files = [os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and f.endswith(".jpg")]
            for file in files:
                # Write the absolute path and subfolder counting to the output file
                f.write("{} {}\n".format(os.path.relpath(file), i))

if __name__ == "__main__":
    # Define the directory path
    dir_path = "data_preprocess/train"
    # Define the output file path
    output_file = "train_label.txt"
    generate_label(dir_path, output_file)

    dir_path = "data_preprocess/val"
    output_file = "val_label.txt"
    generate_label(dir_path, output_file)

    dir_path = "data_preprocess/test"
    output_file = "test_label.txt"
    generate_label(dir_path, output_file)
                   