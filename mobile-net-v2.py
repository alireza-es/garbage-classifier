import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import random
import wandb

# Define the number of classes, batch size, and number of epochs
num_classes = 5  # Replace 5 with your actual number of classes
batch_size = 32  # Example batch size, adjust as needed
num_epochs = 10  # Example number of epochs, adjust as needed
train_data_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset/Train"
valid_data_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset/Validation"
test_data_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset/Test"
# Define the fraction of the dataset you want to use (e.g., 0.05 for 5%)
subset_fraction = 1

# Init wandb
wandb.init(project="ENEL645", entity="far-team", config={
    "num_classes": num_classes,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": 0.001,
    "model_name": "MobileNetV2",
    "dataset_fraction": subset_fraction
})

# Transformations
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # Normalize the image using ImageNet's mean and standard deviation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SubsetImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, subset_fraction=1.0):
        super(SubsetImageFolder, self).__init__(root, transform=transform)
        
        # Calculate the number of images to include from each class
        num_images = int(len(self.imgs) * subset_fraction)
        
        # Randomly select a subset of indices to keep
        self.indices = random.sample(range(len(self.imgs)), num_images)
        
        # Keep only the selected subset of images
        self.imgs = [self.imgs[i] for i in self.indices]
        self.samples = [self.samples[i] for i in self.indices]
        self.targets = [self.targets[i] for i in self.indices]

# Training dataset and dataloader using the subset
train_dataset = SubsetImageFolder(root=train_data_path, transform=data_transforms, subset_fraction=subset_fraction)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and dataloader using the subset
valid_dataset = SubsetImageFolder(root=valid_data_path, transform=data_transforms, subset_fraction=subset_fraction)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Test dataset and dataloader using the subset
test_dataset = SubsetImageFolder(root=test_data_path, transform=data_transforms, subset_fraction=subset_fraction)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def extract_labels_from_filename(filename):
    parts = filename.split('_')[:-1]  # Split by '_' and exclude the last part (number and extension)
    return parts

# Example usage inside your training loop
for inputs, labels in train_loader:
    filenames = [train_dataset.imgs[idx][0].split('/')[-1] for idx in range(len(inputs))]
    additional_labels = [extract_labels_from_filename(filename) for filename in filenames]
    print(additional_labels)  # Do something with these labels


# Correctly set up the model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Adjust the classifier for your number of classes

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move the model to the specified device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
wandb.watch(model, criterion, log="all", log_freq=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Rest of your setup (data loaders, etc.) appears correct

# Update the training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    # Inside your training loop after loss calculation
    wandb.log({"train_loss": train_loss})

    # Validation Loop
    model.eval()  # Set the model to evaluation mode
    valid_loss = 0.0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)

    valid_loss = valid_loss / len(valid_loader.dataset)
    # Inside your validation loop after loss calculation
    wandb.log({"valid_loss": valid_loss})

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

# Testing Loop and any other components follow the same pattern.
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = 100 * correct / total
wandb.log({"test_accuracy": test_accuracy})

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Save the model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'model_checkpoint.pth')

# Load the model checkpoint
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
wandb.finish()
