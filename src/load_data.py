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


def build_tensor_dataset(df: pd.DataFrame, img_size: int = 128, include_images: bool = True, include_features: List[str] = None) -> TensorDataset:
    preprocessing = v2.Compose(
        [
            Image.open,
            v2.PILToTensor(),
            square_image,
            v2.Resize((img_size, img_size)),
        ]
    )

    tensors = []
    images = None
    features = None
    
    if include_images:
        images = torch.stack([preprocessing(image_path) for image_path in df["id"]])
        tensors.append(images)

    if include_features is not None:
        feature_columns = df.columns.drop('id', errors='ignore')
        features_to_include = []
        for feature_category in include_features:
            matched_features = [col for col in feature_columns if feature_category in col]
            features_to_include.extend(matched_features)
        features = torch.tensor(df[features_to_include].values, dtype=torch.float32)
        tensors.append(features)

    if "species" in df.columns:
        labels = torch.tensor(df["species"].values, dtype=torch.int64)
        tensors.append(labels)
    else:
        sample_ids = [int(os.path.splitext(os.path.basename(img_path))[0]) for img_path in df["id"].values]
        sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.int64)
        tensors.append(sample_ids_tensor)

    if not tensors:
        raise ValueError("No valid tensors were created for the TensorDataset.")

    dataset = TensorDataset(*tensors)

    return dataset


def collate_fn_factory(transform, include_images=True, include_features=['margin', 'shape', 'texture']):
    def collate_fn(batch):
        images, features, labels = [], [], []

        for elements in batch:
            if include_images and include_features:
                # Cas où les images et les caractéristiques tabulaires sont présentes
                img, feat, lab = elements
                if include_images:
                    img = transform(img)
                    images.append(img)
                if include_features:
                    features.append(feat)
                labels.append(lab)
            elif include_images:
                # Cas où seulement les images sont présentes
                img, lab = elements
                img = transform(img)
                images.append(img)
                labels.append(lab)
            elif include_features:
                # Cas où seulement les caractéristiques tabulaires sont présentes
                feat, lab = elements
                features.append(feat)
                labels.append(lab)

        # Construire les tenseurs à partir des listes
        tensors = []
        if include_images and images:
            images_tensor = torch.stack(images)
            tensors.append(images_tensor)
        if include_features and features:
            features_tensor = torch.stack(features)
            tensors.append(features_tensor)
        labels_tensor = torch.stack(labels)
        tensors.append(labels_tensor)

        return tuple(tensors)

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
    use_k_fold: bool = True,
    n_splits: int = 5,
    include_images: bool = True,
    include_features: List[str] = ['margin', 'shape', 'texture']
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_df, val_df, test_df, species = load_csv(
        data_dir, test_size=test_size, random_state=random_state
    )

    # Préparation des datasets
    train_dataset = build_tensor_dataset(train_df, img_size, include_images, include_features)
    val_dataset = build_tensor_dataset(val_df, img_size, include_images, include_features)
    test_dataset = build_tensor_dataset(test_df, img_size, include_images, include_features)

    if data_augmentation:
        transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(20, scale=(0.9, 1.05)),
        ])
        collate_fn = collate_fn_factory(transform, include_images=include_images, include_features=include_features)
    else:
        collate_fn = None
    
    if use_k_fold:
        train_df, _, test_df, species = load_csv(data_dir, test_size=None, random_state=random_state)

        full_train_dataset = build_tensor_dataset(train_df, img_size, include_images, include_features)
        
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
