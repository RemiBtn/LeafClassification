from __future__ import annotations
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset


def load_csv(
    data_dir: str,
    test_size: float | int | None = 0.2,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_df["id"] = train_df["id"].apply(
        lambda sample_id: os.path.join(data_dir, "images", f"{sample_id}.jpg")
    )
    test_df["id"] = test_df["id"].apply(
        lambda sample_id: os.path.join(data_dir, "images", f"{sample_id}.jpg")
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["species"])
    species = label_encoder.classes_
    train_df["species"] = label_encoder.transform(train_df["species"])

    train_df, val_df = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df["species"].values,
    )

    return train_df, val_df, test_df, species


def square_image(img: torch.Tensor) -> torch.Tensor:
    img = img[0]
    h, w = img.shape
    pad = abs(w - h) // 2

    if h < w:
        sq_img = torch.zeros(w, w, dtype=torch.float32)
        sq_img[pad : pad + h] = img
    else:
        sq_img = torch.zeros(h, h, dtype=torch.float32)
        sq_img[:, pad : pad + w] = img

    sq_img = sq_img.unsqueeze(0)
    return sq_img


def build_tensor_dataset(df: pd.DataFrame, img_size: int = 128) -> TensorDataset:
    preprocessing = v2.Compose(
        [
            Image.open,
            v2.PILToTensor(),
            square_image,
            v2.Resize((img_size, img_size)),
        ]
    )
    feature_names = (
        [f"margin{i}" for i in range(1, 65)]
        + [f"shape{i}" for i in range(1, 65)]
        + [f"texture{i}" for i in range(1, 65)]
    )

    images = [preprocessing(image_path) for image_path in df["id"]]
    images = torch.stack(images)
    features = torch.tensor(df[feature_names].values, dtype=torch.float32)

    if "species" in df.columns:
        labels = torch.tensor(df["species"].values, dtype=torch.int64)
        dataset = TensorDataset(images, features, labels)
    else:
        samples_ids = [
            int(os.path.splitext(os.path.basename(img_path))[0])
            for img_path in df["id"].values
        ]
        sample_ids = torch.tensor(samples_ids, dtype=torch.int64)
        dataset = TensorDataset(images, features, sample_ids)

    return dataset


def collate_fn_factory(transform):
    def collate_fn(batch):
        images = []
        features = []
        labels = []

        for img, feat, lab in batch:
            img = transform(img)
            images.append(img)
            features.append(feat)
            labels.append(lab)

        images = torch.stack(images)
        features = torch.stack(features)
        labels = torch.stack(labels)

        return images, features, labels

    return collate_fn


def get_data_loaders(
    train_batch_size: int = 64,
    data_augmentation: bool = True,
    img_size: int = 128,
    test_batch_size: int = 1024,
    *,
    data_dir: str = "../data",
    test_size: float | int | None = 0.2,
    random_state: int = 42,
    use_k_fold: bool = False,
    n_splits: int = 5,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_df, val_df, test_df, species = load_csv(
        data_dir, test_size=test_size, random_state=random_state
    )

    # Préparation des datasets
    train_dataset = build_tensor_dataset(train_df, img_size)
    val_dataset = build_tensor_dataset(val_df, img_size)
    test_dataset = build_tensor_dataset(test_df, img_size)

    if data_augmentation:
        transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(20, scale=(0.9, 1.05)),
        ])
        collate_fn = collate_fn_factory(transform)
    else:
        collate_fn = None
    
    if use_k_fold:
        train_df, _, test_df, species = load_csv(data_dir, test_size=None, random_state=random_state)

        full_train_dataset = build_tensor_dataset(train_df, img_size)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_loaders = []
        val_loaders = []

        labels = train_df['species'].values
        
        for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels): 
            train_subset = Subset(full_train_dataset, train_idx)
            val_subset = Subset(full_train_dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
            val_loader = DataLoader(val_subset, batch_size=test_batch_size)
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        # Le DataLoader pour le test reste inchangé
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)
        return train_loaders, val_loaders, test_data_loader, species
    else:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_data_loader = DataLoader(val_dataset, test_batch_size)
        test_data_loader = DataLoader(test_dataset, test_batch_size)
        return train_data_loader, val_data_loader, test_data_loader, species
