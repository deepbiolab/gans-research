"""
Datasets and DataLoaders for PyTorch.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataset(config):
    """
    Create a dataset based on the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        dataset: PyTorch dataset
    """
    dataset_name = config["data"]["name"]
    image_size = config["data"]["image_size"]

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
        dataset = datasets.MNIST(
            root="./datasets", train=True, download=True, transform=transform
        )
    elif dataset_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root="./datasets", train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root="./datasets", train=True, download=True, transform=transform
        )
    elif dataset_name == "celeba":
        # For CelebA, we need center crop before resize
        crop_size = config["data"].get("crop_size", 178)  # Default CelebA crop
        transform = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = datasets.CelebA(
            root="./data/downloaded", split="train", download=True, transform=transform
        )
    elif dataset_name in ["celeba_hq", "ffhq"]:
        # For high-res datasets like CelebA-HQ and FFHQ
        # Note: These datasets require external download and preprocessing
        # This is just a placeholder - you'll need to implement custom dataset classes
        raise NotImplementedError(
            f"Dataset {dataset_name} requires custom implementation and preprocessing. "
            f"Please add your implementation in data/datasets/{dataset_name}.py"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_dataloader(config):
    """
    Create a data loader for the specified dataset.

    Args:
        config: Configuration dictionary

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = get_dataset(config)
    batch_size = config["data"]["batch_size"]
    num_workers = config["experiment"].get("num_workers", 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop the last incomplete batch
    )
