import argparse

import lightning as L
import torch

from datamodules.dogbreed import DogImageDataModule
from models.timm_classifier import DogClassifier
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def main(args):

    # 1. data module
    data_module = DogImageDataModule(num_workers=2, batch_size=16)

    # 2. set up the data module for validation data
    data_module.setup(stage="fit")

    # 3. validation datset
    val_dataset = data_module.val_dataset

    # 4. data loader
    val_data_loader = data_module.val_dataloader()

    # 5. load model
    model = DogClassifier.load_from_checkpoint(args.ckpt_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=1,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # 7. evaluate the module
    results = trainer.test(
        model=model,
        datamodule=data_module,
    )

    log.info("Validation is completed!!!")
    log.info(f"validation results: {results}")


if __name__ == "__main__":

    """
    This function parses command line arguments for the model checkpoint path and calls the main function to perform evaluation on images.

    """

    parser = argparse.ArgumentParser(description="Performs evaluation on images")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="model/dog_breed_classifier_model.ckpt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()
    main(args)
