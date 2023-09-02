# Chinese-Medicine-Classification-System
Undergraduate Final Year Project

This repository contains the code and resources for the Chinese Medicine Classification System, an undergraduate final year project.

## Directory Structure

- `code`: Contains all the files in the project.
    - `data`: Contains two folders, `data` and `test`.
        - `data`: Collection of the train set and val set (which has not been divided).
        - `test`: Test set.
        
    The number of samples in these two directories is 8:2.

    - `SplitDataset.py`: Script used to split the `data/data` folder into a train set and val set with a ratio of 8:2.
    - `ResizeImage.py`: Script used to resize the image file size to (224x224). The processed images will be presented in the `data_preprocess` directory.
    - `DataAugmentation.py`: Script used to augment the image files. The `train` directory in the `data_preprocess` (the train set) will be augmented.
    - `LabelGenerator.py`: Script for generating 3 text files, namely `train_label.txt`, `val_label.txt`, and `test_label.txt`. These text files are the labels for the dataset and will be utilized in the training and testing phase.
    - `Test.py`: Script for testing the model based on the `test_label.txt`.
    - `Train.py`: Script for training the model.

    `resnet34_model.pth`, `resnet50_model.pth`: Model parameters for ResNet-34, ResNet-50, respectively.

- `web`: Contains the files for the web application using Django framework.

## Usage

To reproduce the application, follow these steps:

1. Run `SplitDataset.py`
2. Run `ResizeImage.py`
3. Run `DataAugmentation.py`
4. Run `LabelGenerator.py`
5. Run `Train.py`
6. Run `Test.py`
7. Enter the path `web\myproject` and run the `manage.py` with the following commands: `python manage.py migrate`, `python manage.py runserver`.

To use the application directly, follow the 7th step.

## Environment

- OS: Windows 11
- Python: 3.9.13 (conda 23.1.0)
- IDE: Visual Studio Code

## Packages

| Package                       | Version                   |
|-------------------------------|---------------------------|
| opencv-python                 | 4.7.0.72                  |
| torch                         | 2.0.0                     |
| torchaudio                    | 2.0.0                     |
| torchdata                     | 0.6.0                     |
| torchtext                     | 0.4.0                     |
| torchvision                   | 0.15.0                    |
| tornado                       | 6.1                       |
| scikit-image                  | 0.19.2                    |
| scikit-learn                  | 1.0.2                     |
| scikit-learn-intelex          | 2021.20221004.171935      |
| imgaug                        | 0.4.0                     |
| Pillow                        | 9.2.0                     |
| Django                        | 4.2                       |

