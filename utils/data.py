from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_file: str
    label_file: str
    data_key: str
    label_key: str
    num_classes: int


DATASET_CONFIGS = {
    "HC": DatasetConfig("HC", "holden.mat", "holden_gt.mat", "holden", "holden_gt", 6),
    "NF": DatasetConfig("NF", "NiliFossae.mat", "NiliFossae_gt.mat", "NiliFossae", "NiliFossae_gt", 9),
    "UP": DatasetConfig("UP", "Utopia.mat", "Utopia_gt.mat", "Utopia", "Utopia_gt", 9),
}


def get_dataset_config(dataset_name):
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_CONFIGS:
        valid_names = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid_names}.")
    return DATASET_CONFIGS[dataset_name]


def load_dataset(dataset_name, data_dir):
    config = get_dataset_config(dataset_name)
    data_dir = Path(data_dir)
    data_path = data_dir / config.data_file
    label_path = data_dir / config.label_file

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    data_mat = sio.loadmat(data_path)
    label_mat = sio.loadmat(label_path)
    if config.data_key not in data_mat:
        raise KeyError(f"Key '{config.data_key}' was not found in {data_path}.")
    if config.label_key not in label_mat:
        raise KeyError(f"Key '{config.label_key}' was not found in {label_path}.")

    data = data_mat[config.data_key]
    labels = label_mat[config.label_key]
    return data, labels, config


def apply_pca(data, num_components):
    reshaped = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    transformed = pca.fit_transform(reshaped)
    return np.reshape(transformed, (data.shape[0], data.shape[1], num_components))


def pad_with_zeros(data, margin):
    padded = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]))
    padded[margin : data.shape[0] + margin, margin : data.shape[1] + margin, :] = data
    return padded


def create_image_cubes(data, labels, window_size=13, remove_zero_labels=True):
    margin = int((window_size - 1) / 2)
    padded = pad_with_zeros(data, margin=margin)

    patches = np.zeros((data.shape[0] * data.shape[1], window_size, window_size, data.shape[2]))
    patch_labels = np.zeros((data.shape[0] * data.shape[1]))
    patch_index = 0

    for row in range(margin, padded.shape[0] - margin):
        for col in range(margin, padded.shape[1] - margin):
            patch = padded[row - margin : row + margin + 1, col - margin : col + margin + 1]
            patches[patch_index, :, :, :] = patch
            patch_labels[patch_index] = labels[row - margin, col - margin]
            patch_index += 1

    if remove_zero_labels:
        valid_mask = patch_labels > 0
        patches = patches[valid_mask, :, :, :]
        patch_labels = patch_labels[valid_mask] - 1

    return patches, patch_labels.astype(np.int64)


def split_fixed_train_per_class(data, labels, train_samples_per_class, random_state=42):
    rng = np.random.RandomState(random_state)
    train_indices = []
    test_indices = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) < train_samples_per_class:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} samples, "
                f"which is fewer than train_samples_per_class={train_samples_per_class}."
            )

        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        train_indices.extend(shuffled[:train_samples_per_class])
        test_indices.extend(shuffled[train_samples_per_class:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]


def split_by_test_ratio(data, labels, test_ratio, random_state=42):
    return train_test_split(
        data,
        labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
    )


def prepare_data(data, labels, patch_size=13, use_pca=True, pca_components=30):
    if use_pca:
        data = apply_pca(data, num_components=pca_components)

    cubes, cube_labels = create_image_cubes(data, labels, window_size=patch_size)
    cubes = cubes.transpose(0, 3, 1, 2)
    return cubes, cube_labels, data.shape[2]


class HSIDataset(Dataset):
    def __init__(self, data, labels):
        self.x_data = torch.as_tensor(data, dtype=torch.float32)
        self.y_data = torch.as_tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]
