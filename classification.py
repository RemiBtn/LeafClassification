import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# Load labels from CSV
labels_df = pd.read_csv('data/train.csv')
# Assuming the CSV has 'id' and 'species' columns
labels_df['id'] = labels_df['id'].apply(lambda x: str(x) + '.jpg')  # Adjust file extension if needed

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels_df['species'])  # Fit the encoder to all unique species
label_dict = {row['id']: label_encoder.transform([row['species']])[0] for _, row in labels_df.iterrows()}

# Custom Dataset Class
class LeafDataset(Dataset):
    def __init__(self, image_dir, label_dict=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_dict = label_dict
        if label_dict is not None:
            self.images = list(label_dict.keys())
        else:
            self.images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]  # Adjust file extension if needed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.label_dict is not None:
            label = self.label_dict[img_name]
            return image, label
        else:
            return image

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 600)
        self.fc2 = nn.Linear(600, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        
# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Split data into labeled and unlabeled sets
image_directory = 'data/images'
train_images = list(label_dict.keys())
test_images = [img for img in os.listdir(image_directory) if img not in train_images and img.endswith('.jpg')]  # Adjust file extension if needed

# Further split training images into training and validation sets
train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

# Create label dictionaries for train and validation sets
train_label_dict = {img: label_dict[img] for img in train_images}
val_label_dict = {img: label_dict[img] for img in val_images}

# Create dataset objects
train_dataset = LeafDataset(image_directory, train_label_dict, transform=transform)
val_dataset = LeafDataset(image_directory, val_label_dict, transform=transform)
test_dataset = LeafDataset(image_directory, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, Loss, Optimizer
model = CNN(num_classes=len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop with Validation
num_epochs = 10
for epoch in range(num_epochs):
    # Training loop
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}')

# Load test.csv to get the list of test image IDs
test_csv_path = 'data/test.csv'
test_df = pd.read_csv(test_csv_path)
test_image_ids = test_df['id'].tolist()

# Assuming model and label_encoder are already defined and loaded
model.eval()  # Set the model to evaluation mode

# Initialize list to store DataFrame rows
rows_list = []

# Process each test image and make predictions
for image_id in test_image_ids:
    image_path = os.path.join(image_directory, f"{image_id}.jpg")  # Adjust extension if needed
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = probabilities.cpu().numpy()

    # Prepare row for submission
    row = [image_id] + prediction[0].tolist()
    rows_list.append(row)

# Create DataFrame from the list of rows
species = label_encoder.classes_  # Assuming label_encoder is already defined and fitted
submission_df = pd.DataFrame(rows_list, columns=['id'] + species.tolist())

# Save submission to CSV
submission_csv_path = 'data/submission.csv'
submission_df.to_csv(submission_csv_path, index=False)

submission_csv_path