from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from load_data import get_data_loaders
from models import LightModel, MixedInputModel
from typing import List, Union

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_dirname(
    name: str | None = None, input_type: str | None = "image_only"
) -> str:
    if name is None:
        name = time.strftime("%Y%m%d-%H%M%S")

    if input_type is None:
        dirname = name
    else:
        dirname = os.path.join(input_type, name)

    ext = ""
    cnt = 1
    while os.path.exists(os.path.join("..", "runs", "metrics", dirname + ext)):
        cnt += 1
        ext = f"_{cnt}"

    return dirname + ext


def training_loop(
    model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, criterion, include_images: bool, include_features: List[str]
) -> tuple[float, float]:
    model.train()
    n_samples = 0
    running_loss = 0
    n_correct = 0

    with tqdm.tqdm(
        train_loader,
        desc="Training",
        unit="batch",
        bar_format="{desc}:   {percentage:3.0f}%|{bar:30}| {n:>3d}/{total:>3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as progress_bar:
        for batch in progress_bar:
            optimizer.zero_grad()

            # Prepare data based on input types
            if include_images and include_features is not None:
                images, features, labels = batch
                images = images.to(device)
                features = features.to(device)
            elif include_images:
                images, labels = batch
                images = images.to(device)
                features = None  # No features to process
            elif include_features is not None:
                features, labels = batch
                features = features.to(device)
                images = None  # No images to process
            labels = labels.to(device)

            # Adjust model call based on available data
            if images is not None and features is not None:
                probability_logits = model(image=images, features=features)
            elif images is not None:
                probability_logits = model(image=images)  # Assuming your model can handle this case
            elif features is not None:
                probability_logits = model(features=features)  # Assuming your model can handle this case

            loss = criterion(probability_logits, labels)
            loss.backward()
            optimizer.step()

            _, predicted_labels = torch.max(probability_logits, 1)
            n_samples_batch = labels.size(0)
            n_samples += n_samples_batch
            running_loss += loss.item() * n_samples_batch
            n_correct += (predicted_labels == labels).sum().item()

            postfix_str = f"loss: {running_loss / n_samples:.4f}   accuracy: {100 * n_correct / n_samples:.2f}%"
            progress_bar.set_postfix_str(postfix_str)

    train_loss = running_loss / n_samples
    train_accuracy = 100 * n_correct / n_samples

    return train_loss, train_accuracy


def validation_loop(
    model: nn.Module, val_loader: DataLoader, criterion, include_images: bool, include_features: List[str] or None
) -> tuple[float, float]:
    model.eval()
    n_samples = 0
    running_loss = 0
    n_correct = 0

    with tqdm.tqdm(
        val_loader,
        desc="Validation",
        unit="batch",
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n:>3d}/{total:>3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as progress_bar:
        with torch.no_grad():
            for batch in progress_bar:
                # Adjust the unpacking of batch data based on the presence of images and features
                if include_images and include_features is not None:
                    images, features, labels = batch
                    images = images.to(device)
                    features = features.to(device)
                elif include_images:
                    images, labels = batch
                    images = images.to(device)
                    features = None  # Handle case where features are not used
                elif include_features is not None:
                    features, labels = batch
                    features = features.to(device)
                    images = None  # Handle case where images are not used
                labels = labels.to(device)

                # Adjust the model call based on available data
                if images is not None and features is not None:
                    probability_logits = model(image=images, features=features)
                elif images is not None:
                    probability_logits = model(image=images)  # Ensure your model supports this call
                elif features is not None:
                    probability_logits = model(features=features)  # Ensure your model supports this call

                loss = criterion(probability_logits, labels)
                _, predicted_labels = torch.max(probability_logits, 1)

                n_samples_batch = labels.size(0)
                n_samples += n_samples_batch
                running_loss += loss.item() * n_samples_batch
                n_correct += (predicted_labels == labels).sum().item()

                postfix_str = f"loss: {running_loss / n_samples:.4f}   accuracy: {100 * n_correct / n_samples:.2f}%"
                progress_bar.set_postfix_str(postfix_str)

    val_loss = running_loss / n_samples
    val_accuracy = 100 * n_correct / n_samples

    return val_loss, val_accuracy


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loaders: Union[DataLoader, List[DataLoader]],
    val_loaders: Union[DataLoader, List[DataLoader]],
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    num_epochs: int = 100,
    criterion=nn.CrossEntropyLoss(),
    dirname: str | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    include_images: bool = True, 
    include_features: List[str] = ['margin', 'shape', 'texture']
) -> nn.Module:

    if dirname is None:
        dirname = time.strftime("%Y%m%d-%H%M%S")
    metrics_dir = os.path.join("..", "runs", "metrics", dirname)
    model_path = os.path.join("..", "runs", "models", dirname + ".pth")
    writer = SummaryWriter(metrics_dir)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = model.to(device)
    best_val_accuracy = -1

    if isinstance(train_loaders, DataLoader):
        train_loaders = [train_loaders]
        val_loaders = [val_loaders]

    total_train_loss, total_val_loss, total_train_accuracy, total_val_accuracy = 0, 0, 0, 0

    # for name, param in model.named_parameters():
    #     writer.add_histogram(f'Weights/{name}', param, 0)
    #     if param.requires_grad:
    #         # Les gradients sont None avant la première backward
    #         if param.grad is not None:
    #             writer.add_histogram(f'Gradients/{name}', param.grad, 0)

    for epoch in range(1, num_epochs + 1):
        epoch_train_loss, epoch_train_accuracy = 0.0, 0.0
        epoch_val_loss, epoch_val_accuracy = 0.0, 0.0
        
        for fold, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
            train_loss, train_accuracy = training_loop(
                model, optimizer, train_loader, criterion, include_images=include_images,include_features=include_features)
            epoch_train_loss += train_loss
            epoch_train_accuracy += train_accuracy

            val_loss, val_accuracy = validation_loop(
                model, val_loader, criterion, include_images=include_images, include_features=include_features)
            epoch_val_loss += val_loss
            epoch_val_accuracy += val_accuracy
            
        epoch_train_loss /= len(train_loaders)
        epoch_train_accuracy /= len(train_loaders)
        epoch_val_loss /= len(val_loaders)
        epoch_val_accuracy /= len(val_loaders)
        
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_accuracy, epoch)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", epoch_val_accuracy, epoch)
        
        print(f"Epoch {epoch}/{num_epochs} - Loss Train: {epoch_train_loss:.4f}, Acc Train: {epoch_train_accuracy:.4f}, Loss Val: {epoch_val_loss:.4f}, Acc Val: {epoch_val_accuracy:.4f}")

        # if epoch%20==0:
        #     for name, param in model.named_parameters():
        #         writer.add_histogram(f'Weights/{name}', param, epoch)
        #         if param.requires_grad:
        #             # Assurez-vous que .grad n'est pas None après la backward
        #             if param.grad is not None:
        #                 writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        if scheduler is not None:
            scheduler.step()

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at {epoch} with an accuracy of {best_val_accuracy:.4f}")

    return model


def make_submission_csv(
    model: nn.Module,
    test_loader: DataLoader,
    species: np.ndarray,
    dirname: str | None = None,
    include_images: bool = True,
    include_features: List[str] or None = None
):
    print()
    time.sleep(0.01)  # avoid display issues with tqdm

    columns = ["id"] + species.tolist()
    rows = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(
            test_loader,
            desc="Test",
            unit="batch",
            bar_format="{desc}:       {percentage:3.0f}%|{bar:30}| {n:>3d}/{total:>3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ):
            # Adapt the data unpacking based on what's included
            if include_images and include_features is not None:
                images, features, sample_ids = batch
                images = images.to(device)
                features = features.to(device)
                probability_logits = model(image=images, features=features)
            elif include_images:
                images, sample_ids = batch
                images = images.to(device)
                probability_logits = model(image=images)  # Ensure your model supports this call
            elif include_features is not None:
                features, sample_ids = batch
                features = features.to(device)
                probability_logits = model(features=features)  # Ensure your model supports this call

            probabilities = torch.softmax(probability_logits, dim=1)
            probabilities = probabilities.cpu()

            for sample_id, prediction in zip(sample_ids, probabilities):
                row = [sample_id.item()] + prediction.tolist()
                rows.append(row)

    if dirname is None:
        dirname = time.strftime("%Y%m%d-%H%M%S")
    submission_csv_path = os.path.join("..", "data", "submissions", dirname + ".csv")
    os.makedirs(os.path.dirname(submission_csv_path), exist_ok=True)

    submission_df = pd.DataFrame(rows, columns=columns)
    submission_df.to_csv(submission_csv_path, index=False)

    print(f"Submission saved in {submission_csv_path}.")


def calculate_num_features(include_features: List[str]) -> int:
    features_per_category = {'margin': 64,'shape': 64,'texture': 64 }

    if include_features is None:
        return 0

    num_features = sum(features_per_category[feature] for feature in include_features if feature in features_per_category)
    return num_features


def experiment(name: str | None = None, 
    input_type: str | None = None, 
    train_batch_size: int = 64, 
    data_augmentation: bool = True, 
    use_k_fold: bool = True, 
    n_splits: int = 5, 
    num_epochs: int = 200, 
    include_images: bool = True, 
    include_features: List[str] = ['margin', 'shape', 'texture']):

    dirname = build_dirname(name, input_type)
    train_loader, val_loader, test_loader, species = get_data_loaders(
        train_batch_size=train_batch_size, 
        data_augmentation=data_augmentation, 
        use_k_fold=use_k_fold, 
        n_splits= n_splits,
        include_images=include_images,
        include_features=include_features
        )
    num_features = calculate_num_features(include_features)
    model = LightModel(include_images=include_images,num_features=num_features)
    optimizer = optim.AdamW(model.parameters())
    scheduler = LambdaLR(
        optimizer,
        lambda epoch: (epoch <= 10)
        + 0.4 * (10 < epoch <= 75)
        + 0.08 * (75 < epoch <= 150)
        + 0.02 * (150 < epoch),
    )
    model = train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        num_epochs=num_epochs,
        dirname=dirname,
        include_images=include_images,
        include_features=include_features,
    )
    make_submission_csv(model, test_loader, species, dirname, include_images=include_images, include_features=include_features)

def main():
    experiment(name="1_layer_features+resnet-3_layers", input_type = "image_only", train_batch_size=64, data_augmentation=True, use_k_fold=True, n_splits=5, num_epochs=100, include_features=None)
    experiment(name="1_layer_features+resnet-3_layers", input_type = "features_only", train_batch_size=64, data_augmentation=True, use_k_fold=True, n_splits=5, num_epochs=100,include_images=False ,include_features=['margin', 'shape', 'texture'])
    experiment(name="1_layer_features+resnet-3_layers", input_type = "image+margin", train_batch_size=64, data_augmentation=True, use_k_fold=True, n_splits=5, num_epochs=100, include_features=['margin'])
    experiment(name="1_layer_features+resnet-3_layers", input_type = "image+3features", train_batch_size=64, data_augmentation=True, use_k_fold=True, n_splits=5, num_epochs=100, include_features=['margin', 'shape', 'texture'])

if __name__ == "__main__":
    main()
