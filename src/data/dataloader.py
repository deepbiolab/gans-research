"""
Datasets and DataLoaders for PyTorch (supports GAN/CGAN).
"""

from typing import Tuple, Any, Dict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataset(config: Dict[str, Any], override_image_size: int = None) -> Tuple[Any, Any]:
    """
    Create a dataset based on the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        train_dataset, test_dataset: PyTorch datasets, each returns (img, label)
    """
    dataset_name = config["data"]["name"]
    image_size = override_image_size if override_image_size is not None else config["data"]["image_size"]

    # Define basic transforms
    if config["data"]["channels"] == 1:
        # Grayscale datasets (MNIST, Fashion-MNIST)
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )
    else:
        # RGB datasets
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                ),  # Normalize to [-1, 1]
            ]
        )

    # Create dataset based on name
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root="./datasets", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./datasets", train=False, download=True, transform=transform
        )
    elif dataset_name == "fashion_mnist":
        train_dataset = datasets.FashionMNIST(
            root="./datasets", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./datasets", train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./datasets", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./datasets", train=False, download=True, transform=transform
        )
    elif dataset_name == "celeba":
        # For CelebA, we need center crop before resize
        crop_size = config["data"].get("crop_size")
        transform = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        train_dataset = datasets.CelebA(
            root="./datasets", split="train", download=False, transform=transform
        )
        test_dataset = datasets.CelebA(
            root="./datasets", split="test", download=False, transform=transform
        )
    elif dataset_name in ["celeba_hq", "ffhq"]:
        raise NotImplementedError(
            f"Dataset {dataset_name} requires custom implementation and preprocessing. "
            f"Please add your implementation in src/data/{dataset_name}.py"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset


def create_dataloader(config: Dict[str, Any], override_image_size: int = None):
    """
    Create DataLoaders for training and validation.

    Args:
        config: Configuration dictionary

    Returns:
        train_dataloader, valid_dataloader: Each yields (img, label) tuples for CGAN
    """
    train_dataset, valid_dataset = get_dataset(config, override_image_size)
    batch_size = config["data"]["batch_size"]
    num_workers = config["experiment"].get("num_workers", 4)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_dataloader, valid_dataloader
