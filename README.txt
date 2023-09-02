The followings are explanations of files in the directory named code, which contains all the files in the project.

The data directory contains two folders, named data and test. The folder named data is the collection of the train set and val set (which has not been divided). 
The folder named test is the test set. The number of samples in these two directories is 8:2.

The SplitDataset.py is used to split the data folder in the data directory (data/data) into a train set and val set with a ratio 8:2.

The ResizeImage.py is used to resize the image file size to (224x224). After running the script, the processed image will be presented in the data_preprocess directory.

The DataAumentation.py is used to augment the image files. After running the script, the train directory in the data_preprocess (the train set) will be augmented.

The LabelGenerator.py is for generating 3 text files, namely train_label.txt, val_label.txt, and test_label.txt, 
where these text files are the labels for the dataset and will be utilized in the training and testing phase.

The Test.py is for testing the model based on the test_label.txt. In the main function of Test.py, 
you can change the model you want to test by modifying the statement
 "model = models.resnet50(pretrained=False), or "model = models.resnet34(pretrained=False)" (in line 53 or line 54). 
Also, you need to change the path in line 60 "model.load_state_dict(torch.load(path_of_model))" so as to make the model architecture and model weights consistent.

The Train.py is for training the model. In the main function, you can choose what model you want to train, by modifying the statement in line 106, 107:
"model = models.resnet50(pretrained=True), model = models.resnet34(pretrained=True)".
You can also change the name of the saved model in line 140:
"torch.save(model.state_dict(), 'resnet34_model.pth')"

resnet34_model.pth, resnet50_model.pth are all model parameters, for ResNet-34, ResNet-50, respectively.


The web directory contains the files for the web application using Django framework.
To run the application, enter the path "web\myproject" to run the manage.py with the following command "python manage.py runserver".

In summary: the steps to reproduce the application are
1) Run SplitDataset.py
2) Run ResizeImage.py
3) Run DataAugmentation.py
4) Run LabelGenerator.py
5) Run Train.py
6) Run Test.py
7) Enter the path "web\myproject" to run the manage.py with the following commands "python manage.py migrate", "python manage.py runserver".

where the step to use the application directly is the 7th step.



-----------------------------------------
Environment:

OS: win11
Python: 3.9.13 (conda 23.1.0)
IDE: Visual Studio Code

Package                       Version
----------------------------- --------------------
opencv-python                 4.7.0.72
torch                         2.0.0
torchaudio                    2.0.0
torchdata                     0.6.0
torchtext                     0.4.0
torchvision                   0.15.0
tornado                       6.1
scikit-image                  0.19.2
scikit-learn                  1.0.2
scikit-learn-intelex          2021.20221004.171935
imgaug                        0.4.0
Pillow                        9.2.0
Django                        4.2
