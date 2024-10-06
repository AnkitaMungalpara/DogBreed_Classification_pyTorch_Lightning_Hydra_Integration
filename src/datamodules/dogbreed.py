import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import gdown
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DogImageDataModule(L.LightningModule):
    """
    A PyTorch Lightning DataModule for loading and preparing dog breed images
    for training, validation, and testing. This module manages the datasets
    and handles the dataloader configurations like batch size and number of workers.

    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 4,
        batch_size: int = 8,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        url = "https://drive.google.com/uc?export=download&id=1Bu3HQmZ6_XP-qnEuCVJ4Bg4641JuoPbx"
        file_path = self.data_dir / "data.zip"
        extracted_dir = self.data_dir / "dataset"

        if not extracted_dir.exists():
            # Download and extract only if the dataset doesn't exist
            gdown.download(url, str(file_path), quiet=False)
            with ZipFile(file_path, "r") as file:
                file.extractall(self.data_dir)
            file_path.unlink()  # Remove zip file after extraction

    def split_dataset(self):
        data_path = self.data_dir / "dataset"
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "validation"
        test_path = self.data_dir / "test"

        if not train_path.exists() or not val_path.exists() or not test_path.exists():
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            for class_dir in data_path.iterdir():
                if class_dir.is_dir() and class_dir.name not in ["train", "validation", "test"]:
                    class_train = train_path / class_dir.name
                    class_val = val_path / class_dir.name
                    class_test = test_path / class_dir.name

                    os.makedirs(class_train, exist_ok=True)
                    os.makedirs(class_val, exist_ok=True)
                    os.makedirs(class_test, exist_ok=True)

                    images = [
                        f for f in class_dir.iterdir()
                        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    ]

                    train_val, test = train_test_split(
                        images, test_size=self.test_split, random_state=42
                    )
                    train, val = train_test_split(
                        train_val, 
                        test_size=self.val_split / (self.train_split + self.val_split),
                        random_state=42
                    )

                    for file in train:
                        shutil.move(str(file), str(class_train))
                    for file in val:
                        shutil.move(str(file), str(class_val))
                    for file in test:
                        shutil.move(str(file), str(class_test))

    def setup(self, stage: str):

        self.split_dataset()

        # data_path = self.data_dir / "dataset"

        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                root=self.data_dir / "train", transform=self.train_transform
            )

            self.val_dataset = ImageFolder(
                root=self.data_dir / "validation", transform=self.val_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                root=self.data_dir / "test", transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    @property
    def normalize_transform(self):
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return transform

    @property
    def train_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )
        return transform

    @property
    def val_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

        return transform
