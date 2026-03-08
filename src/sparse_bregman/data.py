# -*- coding: utf-8 -*-

import os
import h5py

import numpy as np
import torch

from torch import Tensor
from random import shuffle

from torch.utils.data import Dataset, DataLoader
from sparse_bregman.utils import load_json_label_file

__all__ = ["ShapeNetPartDataset"]


class ShapeNetPartDataset(Dataset):
    def __init__(
        self,
        path: str | None = None,
        num_points: int = 2048,
        files: list[str] | None = None,
        cache_size: int = 2048,
        normalize: bool = True,
    ):
        super().__init__()

        self.cache = {}
        self.cache_size = cache_size
        self.normalize = normalize
        self.files = [] if files is None else files

        if path is None:
            parts = __file__.split(os.sep)
            idx = parts.index("src")
            path = __file__.split(os.sep)[1 : idx + 1] + ["data"]
            path = os.sep + os.sep.join(path)

        self.path = path
        self.num_points = num_points
        self.path_pattern = (
            self.path + os.sep + "PartAnnotation" + os.sep + "{}" + os.sep + "points"
        )
        self.categories = os.listdir(self.path + os.sep + "PartAnnotation")
        categorymap = (
            self.path + os.sep + "PartAnnotation" + os.sep + "synsetoffset2category.txt"
        )

        with open(categorymap, "r") as f:
            lines = {l.split("\t")[-1].strip(): l.split("\t")[0] for l in f.readlines()}

        self.categorymap = lines

        self.label2index = dict(zip(lines.values(), range(len(lines))))
        self.index2label = {v: k for k, v in self.label2index.items()}

        self.category2index = {
            k: self.label2index[v] for (k, v) in self.categorymap.items()
        }
        self.index2category = {v: k for k, v in self.category2index.items()}

        if len(self.files) == 0:
            for category in self.categories:
                if os.path.isdir(
                    self.path + os.sep + "PartAnnotation" + os.sep + category
                ):
                    self.files += self.extract_category(category=category)

            shuffle(self.files)

    def category_counts(self, normalize: bool = True):
        categories = list(map(lambda x: x[1], self.files))
        num_categories = len(categories)

        return {
            self.index2label[cat]: (
                (categories.count(cat) / num_categories)
                if normalize
                else categories.count(cat)
            )
            for cat in categories
        }

    def extract_category(self, category: str) -> list[str]:
        path = self.path_pattern.format(category)
        filenames = [
            (f"{path}/{p}", self.category2index[category]) for p in os.listdir(path)
        ]
        return filenames

    def push(self, index, data, label):
        self.cache[index] = (data, label)
        return

    def normalize_cloud(self, data):
        # normalize into a sphere with origin (0, 0, 0) and radius 1.
        mean = np.expand_dims(np.mean(data, axis=0), 0)
        std = np.max(np.sqrt(np.sum(data**2, axis=1)), 0)
        data = (data - mean) / std
        return data

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index in self.cache:
            return self.cache[index]

        file, label = self.files[index]
        data = np.loadtxt(file)

        indices = np.random.choice(len(data), self.num_points, replace=True)

        if isinstance(self.num_points, int):
            data = data[indices, :]

        if self.normalize:
            data = self.normalize_cloud(data=data)

        data, label = torch.tensor(data, dtype=torch.float32), torch.tensor(label)

        if len(self.cache) > self.cache_size:
            self.cache.pop(np.random.randint(low=0, high=len(self.files)))

        # self.cache[index] = (data, label)
        self.push(index=index, data=data, label=label)

        return data, label

    def sample(self, num_samples: int = 1, category: int | str = "all"):
        if not isinstance(category, int):
            files = self.files
        else:
            files = list(filter(lambda x: x[1] == category, self.files))

        sample_indices = np.random.randint(low=0, high=len(files), size=(num_samples))
        sample_files = [files[i] for i in sample_indices]
        sample_indices = [self.files.index(f) for f in sample_files]

        return [self[i] for i in sample_indices]

    def split(
        self, train_fraction: float = 0.5, test_fraction: float = 0.5
    ) -> tuple["ShapeNetPartDataset"]:
        num_train = int(len(self.files) * train_fraction)
        num_test = int(len(self.files) * test_fraction)

        train_files, test_files = self.files[:num_train], self.files[-num_test:]

        train_ds, test_ds = (
            ShapeNetPartDatasetV2(
                path=self.path,
                num_points=self.num_points,
                files=test_files,
                cache_size=self.cache_size,
                normalize=self.normalize,
            ),
            ShapeNetPartDatasetV2(
                path=self.path,
                num_points=self.num_points,
                files=test_files,
                cache_size=self.cache_size,
                normalize=self.normalize,
            ),
        )

        return train_ds, test_ds

    def __len__(self):
        return len(self.files)


class ShapeNetPartDatasetV2(Dataset):
    def __init__(
        self,
        path: str | None = None,
        num_points: int = 2048,
        files: list[str] | None = None,
        cache_size: int = 2048,
        normalize: bool = True,
        data_split: str = "train",
    ):
        super().__init__()

        self.data_split = data_split
        self.normalize = normalize
        self.cache_size = cache_size

        assert num_points == 2048, f"num_points MUST be 2048, not {num_points}."

        self.num_points = num_points

        if path is None:
            parts = __file__.split(os.sep)
            idx = parts.index("src")
            path = __file__.split(os.sep)[1 : idx + 1] + [
                "data",
                "shapenetpart_hdf5_2048",
            ]
            path = os.sep + os.sep.join(path)

        self.path = path

        files = [
            f"{path}/{f}"
            for f in filter(lambda x: x.__contains__(data_split), os.listdir(self.path))
        ]

        data_files = [f for f in files if f.endswith(".h5")]
        label_files = [f for f in files if f.endswith(".json") and "id2name" in f]

        self.data = np.concatenate([h5py.File(f)["data"] for f in data_files])
        self.labels = np.concatenate([h5py.File(f)["label"] for f in data_files])

        unique_labels = set()

        for f in label_files:
            unique_labels.update(set(load_json_label_file(f)))

        unique_labels = sorted(list(unique_labels))

        self.label2index = dict(zip(unique_labels, range(len(unique_labels))))
        self.index2label = {v: k for k, v in self.label2index.items()}

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]
        if self.normalize:
            self.data = self.normalize_cloud(data=self.data)

        return torch.from_numpy(data), torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def normalize_cloud(self, data):
        # normalize into a sphere with origin (0, 0, 0) and radius 1.
        mean = np.expand_dims(np.mean(data, axis=0), 0)
        std = np.max(np.sqrt(np.sum(data**2, axis=1)), 0)
        data = (data - mean) / std
        return data

    def sample(self, num_samples: int = 1, category: int | str = "all"):
        if not isinstance(category, int):
            files = self.data
            labels = self.labels
        else:
            print((self.labels == category).sum())
            files = self.data[(self.labels == category).flatten()]
            labels = self.labels[(self.labels == category).flatten()]

        files = torch.from_numpy(files)
        labels = torch.from_numpy(labels)

        sample_indices = np.random.randint(low=0, high=len(files), size=(num_samples))

        return [(files[i], labels[i]) for i in sample_indices]

    def category_counts(self, normalize: bool = True):
        categories = np.unique(self.labels)
        labels_list = self.labels.tolist()
        num_categories = len(labels_list)

        return {
            self.index2label[cat]: (
                (labels_list.count(cat) / num_categories)
                if normalize
                else labels_list.count(cat)
            )
            for cat in categories
        }

    def split(
        self, train_fraction: float = 0.5, test_fraction: float = 0.5
    ) -> tuple["ShapeNetPartDatasetV2"]:
        train_ds, test_ds = (
            ShapeNetPartDatasetV2(
                path=self.path,
                num_points=self.num_points,
                files=None,
                cache_size=self.cache_size,
                normalize=self.normalize,
                data_split="train",
            ),
            ShapeNetPartDatasetV2(
                path=self.path,
                num_points=self.num_points,
                files=None,
                cache_size=self.cache_size,
                normalize=self.normalize,
                data_split="test",
            ),
        )

        return train_ds, test_ds


if __name__ == "__main__":
    dataset = ShapeNetPartDatasetV2()
    # print(dataset.files)
    # print(len(dataset.files))

    # dataset, dataset_ = dataset.split()

    # print(len(dataset.files))

    X, y = dataset[0]

    print(X)
    print(y)
