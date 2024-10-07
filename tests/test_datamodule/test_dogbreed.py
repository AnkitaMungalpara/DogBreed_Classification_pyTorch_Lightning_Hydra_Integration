import pytest
from pathlib import Path

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datamodules.dogbreed import DogImageDataModule

@pytest.fixture
def datamodule():
    data_dir = root / "data" / "dogbreed"
    return DogImageDataModule(data_dir=data_dir, batch_size=32)

def test_dogbreed_datamodule_init(datamodule):
    assert isinstance(datamodule, DogImageDataModule)
    assert datamodule.data_dir == root / "data" / "dogbreed"
    assert datamodule.batch_size == 32

def test_dogbreed_datamodule_setup(datamodule):
    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    
    datamodule.setup(stage="test")
    assert datamodule.test_dataset is not None

def test_dogbreed_datamodule_dataloaders(datamodule):
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    batch = next(iter(train_loader))
    assert len(batch) == 2  # (images, labels)
    assert batch[0].shape[1:] == (3, 224, 224)  # (batch_size, channels, height, width)