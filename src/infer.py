import argparse
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from datamodules.dogbreed import DogImageDataModule
from models.timm_classifier import DogClassifier

# Define class labels
CLASS_LABELS = [
    "Beagle",
    "Boxer",
    "Bulldog",
    "Dachshund",
    "German_Shepherd",
    "Golden_Retriever",
    "Labrador_Retriever",
    "Poodle",
    "Rottweiler",
    "Yorkshire_Terrier",
]


def denormalize(tensor, mean, std):
    # Ensure tensor is on CPU and in the correct shape (C, H, W)
    tensor = tensor.cpu()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)

    # Reshape mean and std to (C, 1, 1) for broadcasting
    mean = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)

    # Apply denormalization
    return (tensor * std + mean).clamp(0, 1)


def inference(model: pl.LightningModule, img: torch.Tensor) -> Tuple[str, float]:
    """
    Perform inference on a given image using a trained model.

    Args:
        model (pl.LightningModule): Trained PyTorch Lightning model.
        img (torch.Tensor): Input image tensor.

    Returns:
        Tuple[str, float]: predicted label, and confidence.
    """

    # Set the model in evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img)
        probability = F.softmax(output, dim=1)
        predicted = torch.argmax(probability, dim=1).item()

    predicted_label = CLASS_LABELS[predicted]
    confidence = probability[0][predicted].item()

    return predicted_label, confidence


def save_prediction(
    img: torch.Tensor,
    actual_label: str,
    predicted_label: str,
    confidence: float,
    output_path: str,
):
    """
    Save an image with actual and predicted labels, along with confidence.

    Args:
        img (torch.Tensor): The image tensor to be displayed and saved.
        actual_label (str): The ground truth label of the image.
        predicted_label (str): The label predicted by the model.
        confidence (float): The confidence score of the prediction.
        output_path (str): The path where the image with annotations will be saved.
    """

    # Denormalize the image
    img = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Convert the tensor to numpy array
    # From (C, H, W) to (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(9, 9))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Actual: {actual_label} | Predicted: {predicted_label} | (Confidence: {confidence:.2f})"
    )
    plt.savefig(output_path)
    plt.close()


def main(args):
    """
    Main function to load a trained model and perform inference on sample images.

    Args:
        args: Command-line arguments parsed by argparse.
    """

    # Load model
    model = DogClassifier.load_from_checkpoint(args.ckpt_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create a directory for storing predictions if not exists
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. Initialize the data module
    data_module = DogImageDataModule(num_workers=2, batch_size=16)

    # 2. Set up the data module for validation data
    data_module.setup(stage="fit")

    # 3. Retrieve the validation dataset
    val_dataset = data_module.val_dataset

    # Get the indices for sampling
    num_samples = min(args.num_samples, len(val_dataset))  # Limit to available images
    sampled_indices = random.sample(range(len(val_dataset)), num_samples)

    for idx in sampled_indices:
        img, label_index = val_dataset[idx]  # Get the image and its label index
        img_tensor = img.unsqueeze(0).to(model.device)

        # Convert label index to actual label
        actual_label = CLASS_LABELS[label_index]

        predicted_label, confidence = inference(model, img_tensor)

        # Saving the prediction image
        output_image_path = os.path.join(
            args.output_folder, f"sample_{idx}_prediction.png"
        )

        save_prediction(
            img, actual_label, predicted_label, confidence, output_image_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Classification Inference")

    parser.add_argument(
        "--output_folder",
        type=str,
        default="predictions",
        help="Path to save prediction images",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="model/dog_breed_classifier_model.ckpt",
        help="path to the model checkpoint",
    )

    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to process"
    )

    args = parser.parse_args()
    main(args)
