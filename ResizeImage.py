import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                try:
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_folder)
                    output_path = os.path.join(output_folder, relative_path, file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with Image.open(input_path) as img:
                        img = img.resize(size)
                        img.save(output_path)
                        print(f"{file} resized and saved to {output_path}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")


if __name__ == "__main__":
    size = (224, 224)

    input_folder = 'data/test'
    output_folder = 'data_preprocess/test'
    resize_images(input_folder, output_folder, size)

    input_folder = 'data/train'
    output_folder = 'data_preprocess/train'
    resize_images(input_folder, output_folder, size)

    input_folder = 'data/val'
    output_folder = 'data_preprocess/val'
    resize_images(input_folder, output_folder, size)