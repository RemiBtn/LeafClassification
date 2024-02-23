import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from load_data import get_data_loaders
from models import MixedInputModel

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

    return dirname


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
    /,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    *,
    criterion=nn.CrossEntropyLoss(),
    dirname: str | None = None,
) -> nn.Module:
    if dirname is None:
        dirname = time.strftime("%Y%m%d-%H%M%S")
    metrics_dir = os.path.join("..", "runs", "metrics", dirname)
    model_path = os.path.join("..", "runs", "models", dirname + ".pth")
    writer = SummaryWriter(metrics_dir)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = model.to(device)
    best_val_accuracy = -1

    for epoch in range(1, num_epochs + 1):
        print()
        print(f"Epoch {epoch:>3d}/{num_epochs}")
        time.sleep(0.01)  # avoid display issues with tqdm

        train_loss, train_accuracy = training_loop(
            model, optimizer, train_loader, criterion
        )
        val_loss, val_accuracy = validation_loop(model, val_loader, criterion)

        writer.add_scalar("Training Loss", train_loss, epoch)
        writer.add_scalar("Validation Loss", val_loss, epoch)
        writer.add_scalar("Training Accuracy", train_accuracy, epoch)
        writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Saved model at epoch {epoch}.")

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
    train_loader, val_loader, test_loader, species = get_data_loaders()
    model = MixedInputModel()
    optimizer = optim.AdamW(model.parameters())
    model = train_model(model, optimizer, train_loader, val_loader, dirname=dirname)
    make_submission_csv(model, test_loader, species, dirname)


if __name__ == "__main__":
    main()
