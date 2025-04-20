"""Feature extraction utilities for metrics calculation."""

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3Features(nn.Module):
    """
    Inception v3 model for FID score calculation.
    Extracts features from a specific layer of the network.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained Inception model
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.eval()
        # We don't need the classification part
        self.inception.fc = nn.Identity()
        # No gradient computation needed
        for param in self.inception.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extract features for FID calculation.

        Args:
            x: Input images (B, C, H, W) in range [-1, 1]

        Returns:
            features: Extracted features
        """
        # Resize input to the required size for Inception v3
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = nn.functional.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Inception expects images in range [-1, 1]
        # If input is not in this range, normalize (but our GAN outputs are already in [-1, 1])

        # Extract features
        features = self.inception(x)

        return features


def extract_features_for_real_images(
    feature_extractor, dataloader, device, num_samples=None, desc="Extracting features"
):
    """
    Extract features from images using a feature extractor.

    Args:
        feature_extractor: Model to extract features from images
        dataloader: DataLoader providing images
        device: Device to use for computation
        num_samples: Number of samples to extract features from, or None for all
        desc: Description for the progress bar

    Returns:
        features: Numpy array of extracted features
    """
    batch_size = dataloader.batch_size
    if num_samples is not None and batch_size is not None:
        total = int(np.ceil(num_samples / batch_size)) - 1
    else:
        total = None

    features_list = []
    samples_seen = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=total, desc=desc):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)

            # Convert to RGB if necessary
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            # Extract features
            batch_features = feature_extractor(images)
            features_list.append(batch_features.cpu().numpy())

            samples_seen += images.size(0)
            if num_samples is not None and samples_seen >= num_samples:
                break

    # Concatenate all features
    features = np.concatenate(features_list, axis=0)

    # Limit to requested number of samples
    if num_samples is not None:
        features = features[:num_samples]

    return features


def extract_features_for_fake_images(
    feature_extractor,
    generator,
    device,
    num_samples,
    batch_size,
    latent_dim,
    desc="Generating features",
):
    """
    Extract features from generated images.

    Args:
        feature_extractor: Model to extract features from images
        generator: Generator model
        device: Device to use for computation
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        latent_dim: Dimensionality of latent space
        desc: Description for the progress bar

    Returns:
        features: Numpy array of extracted features
    """
    generator = generator.to(device)
    generator.eval()

    features_list = []
    samples_generated = 0

    with torch.no_grad():

        batches = range(0, num_samples, batch_size)
        for _ in tqdm(batches, total=len(batches) - 1, desc=desc):
            current_batch_size = min(batch_size, num_samples - samples_generated)

            # Generate latent vectors
            z = torch.randn(current_batch_size, latent_dim).to(device)

            # Generate images
            fake_images = generator(z)

            # Convert to RGB if necessary
            if fake_images.shape[1] == 1:
                fake_images = fake_images.repeat(1, 3, 1, 1)

            # Extract features
            batch_features = feature_extractor(fake_images)
            features_list.append(batch_features.cpu().numpy())

            samples_generated += current_batch_size
            if samples_generated >= num_samples:
                break

    # Concatenate all features
    features = np.concatenate(features_list, axis=0)

    return features


def extract_features(model, real_dataloader, config):
    """
    Extract features for real and generated images.

    Args:
        model: GAN model
        config: Configuration dictionary
    Returns:
        real_features: Features of real images
        fake_features: Features of generated images
    """
    device = torch.device(config["experiment"]["device"])
    latent_dim = config["model"]["latent_dim"]
    num_samples = config["evaluation"]["num_samples"]
    batch_size = config["evaluation"]["batch_size"]

    # Create feature extractor (reuse the InceptionV3 model from FID)
    feature_extractor = InceptionV3Features().to(device)

    # Extract features from real images
    real_features = extract_features_for_real_images(
        feature_extractor=feature_extractor,
        dataloader=real_dataloader,
        device=device,
        num_samples=num_samples,
    )

    # Extract features from generated images
    fake_features = extract_features_for_fake_images(
        feature_extractor=feature_extractor,
        generator=model.generator,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        latent_dim=latent_dim,
    )

    return real_features, fake_features
