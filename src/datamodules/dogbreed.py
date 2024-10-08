import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile
from torchvision import transforms, datasets
import gdown
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,random_split
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
        file_path = self.data_dir.parent / "data.zip"
        extracted_dir = self.data_dir 
        print("extracted_dir", extracted_dir  )
        print("file_path", file_path)
        if not extracted_dir.exists():
            # Download and extract only if the dataset doesn't exist
            gdown.download(url, str(file_path), quiet=False)
            with ZipFile(file_path, "r") as file:
                file.extractall(self.data_dir)
            file_path.unlink()  # Remove zip file after extraction
        else:
            print("Dataset already exists")

    
    def setup(self, stage: str):

        # Create splits from a single directory
        full_dataset = datasets.ImageFolder(root=self.data_dir/"dataset", transform=self.train_transform)
        print(full_dataset.classes)
        train_size = int(self.train_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        
        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        if stage == "test" or stage is None:
            self.test_dataset = test_dataset
        
        self.class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None
        print("+"*50)
        print(self.class_names)
        print("+"*50)

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