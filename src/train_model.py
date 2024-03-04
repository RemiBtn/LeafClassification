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
    model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, criterion
) -> tuple[float, float]:
    model.train()
    n_samples = 0
    running_loss = 0
    n_correct = 0

    with tqdm.tqdm(
        train_loader,
        desc="Training",
        unit="batch",
        bar_format="{desc}:   {percentage:3.0f}%|"
        "{bar:30}"
        "| {n:>3d}/{total:>3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as progress_bar:
        for images, features, labels in progress_bar:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            probability_logits = model(images, features)
            loss = criterion(probability_logits, labels)
            _, predicted_labels = torch.max(probability_logits, 1)

            n_samples += images.size(0)
            running_loss += loss.item() * images.size(0)
            n_correct += (predicted_labels == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            postfix_str = (
                f"loss: {running_loss / n_samples:.4f}   "
                f"accuracy: {100 * n_correct / n_samples:.2f}%"
            )
            progress_bar.set_postfix_str(postfix_str)

    train_loss = running_loss / n_samples
    train_accuracy = 100 * n_correct / n_samples

    return train_loss, train_accuracy


def validation_loop(
    model: nn.Module, val_loader: DataLoader, criterion
) -> tuple[float, float]:
    model.eval()
    n_samples = 0
    running_loss = 0
    n_correct = 0

    with tqdm.tqdm(
        val_loader,
        desc="Validation",
        unit="batch",
        bar_format="{desc}: {percentage:3.0f}%|"
        "{bar:30}"
        "| {n:>3d}/{total:>3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as progress_bar:
        with torch.no_grad():
            for images, features, labels in progress_bar:
                images = images.to(device)
                features = features.to(device)
                labels = labels.to(device)

                probability_logits = model(images, features)
                loss = criterion(probability_logits, labels)
                _, predicted_labels = torch.max(probability_logits, 1)

                n_samples += images.size(0)
                running_loss += loss.item() * images.size(0)
                n_correct += (predicted_labels == labels).sum().item()

                postfix_str = (
                    f"loss: {running_loss / n_samples:.4f}   "
                    f"accuracy: {100 * n_correct / n_samples:.2f}%"
                )
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
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    for fold, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        fold_train_loss, fold_val_loss, fold_train_accuracy, fold_val_accuracy = 0, 0, 0, 0
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = training_loop(
                model, optimizer, train_loader, criterion
            )
            val_loss, val_accuracy = validation_loop(
                model, val_loader, criterion
            )

            fold_train_loss += train_loss
            fold_train_accuracy += train_accuracy
            fold_val_loss += val_loss
            fold_val_accuracy += val_accuracy
            
            if scheduler is not None:
                scheduler.step()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), model_path)

        # Calcul et enregistrement des moyennes du pli
        num_epochs_float = float(num_epochs)
        fold_train_loss /= num_epochs_float
        fold_train_accuracy /= num_epochs_float
        fold_val_loss /= num_epochs_float
        fold_val_accuracy /= num_epochs_float

        total_train_loss += fold_train_loss
        total_train_accuracy += fold_train_accuracy
        total_val_loss += fold_val_loss
        total_val_accuracy += fold_val_accuracy

        writer.add_scalar(f"Fold_{fold+1}/Average Training Loss", fold_train_loss, fold+1)
        writer.add_scalar(f"Fold_{fold+1}/Average Training Accuracy", fold_train_accuracy, fold+1)
        writer.add_scalar(f"Fold_{fold+1}/Average Validation Loss", fold_val_loss, fold+1)
        writer.add_scalar(f"Fold_{fold+1}/Average Validation Accuracy", fold_val_accuracy, fold+1)

    # Calcul des moyennes globales sur tous les plis
    num_folds_float = float(len(train_loaders))
    avg_train_loss = total_train_loss / num_folds_float
    avg_train_accuracy = total_train_accuracy / num_folds_float
    avg_val_loss = total_val_loss / num_folds_float
    avg_val_accuracy = total_val_accuracy / num_folds_float

    writer.add_scalar("Global/Average Training Loss", avg_train_loss, 0)
    writer.add_scalar("Global/Average Training Accuracy", avg_train_accuracy, 0)
    writer.add_scalar("Global/Average Validation Loss", avg_val_loss, 0)
    writer.add_scalar("Global/Average Validation Accuracy", avg_val_accuracy, 0)

    best_model_state = torch.load(model_path)
    model.load_state_dict(best_model_state)

    return model


def make_submission_csv(
    model: nn.Module,
    test_loader: DataLoader,
    species: np.ndarray,
    dirname: str | None = None,
):
    print()
    time.sleep(0.01)  # avoid display issues with tqdm

    columns = ["id"] + species.tolist()
    rows = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for images, features, sample_ids in tqdm.tqdm(
            test_loader,
            desc="Test",
            unit="batch",
            bar_format="{desc}:       {percentage:3.0f}%|"
            "{bar:30}"
            "| {n:>3d}/{total:>3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ):
            images = images.to(device)
            features = features.to(device)

            probability_logits = model(images, features)
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


def main():
    name = "1_layer_features+resnet-3_layers"
    input_type = "image_and_features"

    dirname = build_dirname(name, input_type)
    train_loader, val_loader, test_loader, species = get_data_loaders(32, use_k_fold=True, n_splits= 5)
    model = LightModel()
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
        num_epochs=25,
        dirname=dirname,
    )
    make_submission_csv(model, test_loader, species, dirname)


if __name__ == "__main__":
    main()
