import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

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


# Define the training loop
def train(model, train_loader, criterion, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)

    return avg_loss, train_accuracy


def validate(model, val_loader, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(val_loader)

    return avg_loss, val_accuracy


if __name__ == "__main__":
    # ... (device, model, dataset, train_set, val_set, train_loader, val_loader, criterion, optimizer remain unchanged) ...
# Use GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the ResNet50 or ResNet34 model and move it to the device
    # model = models.resnet50(pretrained=True)
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    # model.load_state_dict(torch.load('resnet50_model_4__23_7.pth'))
    model = model.to(device)

    train_set = CustomDataset('train_label.txt')
    val_set = CustomDataset('val_label.txt')

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Initialize lists to store the training and validation accuracy values
    train_accuracies = []
    val_accuracies = []

    # Train the model
    for epoch in range(20):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}, Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}, Val Accuracy: {val_accuracy:.2f}')

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Save the trained model
    torch.save(model.state_dict(), 'resnet34_model.pth')

    # Plot the training and validation accuracy
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
