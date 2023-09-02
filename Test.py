import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Define a custom dataset class for loading the images and labels from the label.txt file
class CustomDataset(Dataset):
    def __init__(self, label_file):
        with open(label_file, 'r') as f:
            data = f.readlines()
        
        self.images = []
        self.labels = []
        
        for line in data:
            image_path, label = line.split(' ')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.labels.append(int(label))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        image = self.transform(image)
        
        return image, label



if __name__ == "__main__":
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the ResNet34 or ResNet50 model and move it to the device
    # model = models.resnet34(pretrained=False)
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model = model.to(device)

    # Load the trained model
    model.load_state_dict(torch.load('resnet50_model.pth'))
    model.eval()

    # Load the test dataset
    dataset = CustomDataset('test_label.txt')
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, timeout=30)

    # Define the criterion for calculating loss
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Calculate the loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Calculate the number of correct predictions
        running_corrects += torch.sum(preds == labels.data)

        # Store predictions and labels for later use in metrics calculations
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    # Calculate the test accuracy and loss
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    # Calculate recall, precision, and F1 score
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print('Test Loss: {:.4f} Test Acc: {:.4f}'.format(test_loss, test_acc))
    print('Recall: {:.4f} Precision: {:.4f} F1 Score: {:.4f}'.format(recall, precision, f1))



