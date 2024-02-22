import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load labels from CSV
labels_df = pd.read_csv("data/train.csv")
# Assuming the CSV has 'id' and 'species' columns
labels_df["id"] = labels_df["id"].apply(
    lambda x: str(x) + ".jpg"
)  # Adjust file extension if needed

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels_df["species"])  # Fit the encoder to all unique species
label_dict = {
    row["id"]: label_encoder.transform([row["species"]])[0]
    for _, row in labels_df.iterrows()
}


# Custom Dataset Class
class LeafDataset(Dataset):
    def __init__(self, image_dir, label_dict=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_dict = label_dict
        if label_dict is not None:
            self.images = list(label_dict.keys())
        else:
            self.images = [
                img for img in os.listdir(image_dir) if img.endswith(".jpg")
            ]  # Adjust file extension if needed

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


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 99),
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


def square_image(img):
    img = img[0]
    h, w = img.shape
    pad = abs(w - h) // 2
    if h < w:
        dst = torch.zeros(w, w, dtype=torch.float32)
        dst[pad : pad + h] = img
    else:
        dst = torch.zeros(h, h, dtype=torch.float32)
        dst[:, pad : pad + w] = img
    return dst.unsqueeze(0)


# Transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        square_image,
        transforms.Resize((128, 128)),
    ]
)

# Split data into labeled and unlabeled sets
image_directory = "data/images"
train_images = list(label_dict.keys())
test_images = [
    img
    for img in os.listdir(image_directory)
    if img not in train_images and img.endswith(".jpg")
]

# Further split training images into training and validation sets
train_images, val_images = train_test_split(
    train_images,
    test_size=0.2,
    random_state=42,
    stratify=label_encoder.transform(labels_df["species"]),
)

# Create label dictionaries for train and validation sets
train_label_dict = {img: label_dict[img] for img in train_images}
val_label_dict = {img: label_dict[img] for img in val_images}

# Create dataset objects
train_dataset = LeafDataset(image_directory, train_label_dict, transform=transform)
val_dataset = LeafDataset(image_directory, val_label_dict, transform=transform)
test_dataset = LeafDataset(image_directory, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model, Loss, Optimizer
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.1)

writer = SummaryWriter("runs/cnn+features/" + time.strftime("%Y%m%d-%H%M%S"))

num_epochs = 50
for epoch in range(num_epochs):
    # Training loop
    model.train()
    correct_predictions_train = 0
    train_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_predictions_train += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_accuracy = correct_predictions_train / len(train_dataset)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataset)}, Train accuracy: {train_accuracy}"
    )

    # Validation loop
    model.eval()
    correct_predictions_val = 0
    val_loss = 0
    best_val_accuracy = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions_val += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_accuracy = correct_predictions_val / len(val_dataset)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_dataset)}, Validation accuracy: {val_accuracy}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict().copy()
        torch.save(model.state_dict(), f"best_model.pth")
        print(f"Saved model of epoch {epoch}")

    writer.add_scalar("Training Loss", train_loss / len(train_dataset), epoch)
    writer.add_scalar("Validation Loss", val_loss / len(val_dataset), epoch)
    writer.add_scalar("Training Accuracy", train_accuracy, epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

# Load test.csv to get the list of test image IDs
test_csv_path = "data/test.csv"
test_df = pd.read_csv(test_csv_path)
test_image_ids = test_df["id"].tolist()

# Assuming model and label_encoder are already defined and loaded
model.eval()  # Set the model to evaluation mode

# Initialize list to store DataFrame rows
rows_list = []

# Use the best model weights for test set
model.load_state_dict(torch.load("best_model.pth"))

# Process each test image and make predictions
for image_id in test_image_ids:
    image_path = os.path.join(
        image_directory, f"{image_id}.jpg"
    )  # Adjust extension if needed
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = probabilities.cpu().numpy()

    # Prepare row for submission
    row = [image_id] + prediction[0].tolist()
    rows_list.append(row)

# Create DataFrame from the list of rows
species = label_encoder.classes_  # Assuming label_encoder is already defined and fitted
submission_df = pd.DataFrame(rows_list, columns=["id"] + species.tolist())

# Save submission to CSV
submission_csv_path = "data/submission.csv"
submission_df.to_csv(submission_csv_path, index=False)
print("Submission done")
