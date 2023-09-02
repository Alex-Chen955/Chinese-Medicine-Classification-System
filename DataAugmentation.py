import imgaug.augmenters as iaa
import cv2
import os

def image_augmentation(input_folder):
    # Define augmentation sequence
    seq1 = iaa.Sequential([
        iaa.Affine(rotate=(-45, 45)),  # rotate the image
    ])

    seq2 = iaa.Sequential([
        iaa.Fliplr(p=0.5),  # horizontal flip
    ])

    seq3 = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 3.0)),  # apply gaussian blur to the image
    ])

    seq4 = iaa.Sequential([
        iaa.Affine(scale=(0.5, 1.5)),  # scale the image
    ])

    seq5 = iaa.Sequential([
        iaa.Crop(percent=(0, 0.2)),  # crop a random part of the image
    ])

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith(".png"):
                try:
                    input_path = os.path.join(root, file)
                    image = cv2.imread(input_path)
                    # Apply augmentation sequence to the image
                    augmented_image1 = seq1(image=image)
                    augmented_image2 = seq2(image=image)
                    augmented_image3 = seq3(image=image)
                    augmented_image4 = seq4(image=image)
                    augmented_image5 = seq5(image=image)

                    # Save augmented image
                    output_path1 = os.path.join(root, f'augmented1_{file}')
                    output_path2 = os.path.join(root, f'augmented2_{file}')
                    output_path3 = os.path.join(root, f'augmented3_{file}')
                    output_path4 = os.path.join(root, f'augmented4_{file}')
                    output_path5 = os.path.join(root, f'augmented5_{file}')

                    cv2.imwrite(output_path1, augmented_image1)
                    cv2.imwrite(output_path2, augmented_image2)
                    cv2.imwrite(output_path3, augmented_image3)
                    cv2.imwrite(output_path4, augmented_image4)
                    cv2.imwrite(output_path5, augmented_image5)

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    input_folder = "data_preprocess/train"
    image_augmentation(input_folder)
