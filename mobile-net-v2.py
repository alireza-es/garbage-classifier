import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Define the number of classes, batch size, and number of epochs
num_classes = 5  # Replace 5 with your actual number of classes
batch_size = 32  # Example batch size, adjust as needed
num_epochs = 10  # Example number of epochs, adjust as needed
train_data_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset/Train"
valid_data_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset/Validation"
test_data_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset/Test"

# Transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Training dataset and dataloader
train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and dataloader
valid_dataset = datasets.ImageFolder(root=valid_data_path, transform=data_transforms)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Test dataset and dataloader
test_dataset = datasets.ImageFolder(root=test_data_path, transform=data_transforms)
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
model.to(device)  # Move the model to the specified device

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
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
