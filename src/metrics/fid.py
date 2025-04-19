"""Utilities for calculating Fréchet Inception Distance (FID) between real and generated images."""

import numpy as np
from scipy import linalg
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


def calculate_fid(real_features, fake_features):
    """
    Calculate FID score between real and fake image features.

    Args:
        real_features: Features from real images (numpy array)
        fake_features: Features from fake images (numpy array)

    Returns:
        fid_score: Fréchet Inception Distance
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID score
    # The formula is: ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2*sqrt(sigma_1*sigma_2))

    # Part 1: Mean squared difference
    mean_diff_squared = np.sum((mu_real - mu_fake) ** 2)

    # Part 2: Trace term
    # We need to handle numerical issues when computing sqrt(sigma_1*sigma_2)
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)

    # FID score
    fid_score = mean_diff_squared + trace_term

    return float(fid_score)


class FIDCalculator:
    """
    Calculator for Fréchet Inception Distance between real and generated images.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.feature_extractor = InceptionV3Features().to(self.device)
        self.feature_extractor.eval()

    def extract_features(self, dataloader, num_samples=None):
        """
        Extract features from images using the Inception v3 model.

        Args:
            dataloader: DataLoader providing images
            num_samples: Number of samples to extract features from, or None for all

        Returns:
            features: Numpy array of extracted features
        """
        features_list = []
        samples_seen = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(self.device)
                else:
                    images = batch.to(self.device)

                # convert to RGB if necessary
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

                # Extract features
                batch_features = self.feature_extractor(images)
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

    def extract_features_from_generator(
        self, generator, num_samples=50000, batch_size=50, latent_dim=100, device=None
    ):
        """
        Extract features from generated images.

        Args:
            generator: Generator model
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            latent_dim: Dimensionality of latent space
            device: Device to use

        Returns:
            features: Numpy array of extracted features
        """
        device = device if device is not None else self.device
        generator = generator.to(device)
        generator.eval()

        features_list = []
        samples_generated = 0

        with torch.no_grad():
            while samples_generated < num_samples:
                current_batch_size = min(batch_size, num_samples - samples_generated)

                # Generate latent vectors
                z = torch.randn(current_batch_size, latent_dim).to(device)

                # Generate images
                fake_images = generator(z)

                if fake_images.shape[1] == 1:
                    fake_images = fake_images.repeat(1, 3, 1, 1)

                # Extract features
                batch_features = self.feature_extractor(fake_images)
                features_list.append(batch_features.cpu().numpy())

                samples_generated += current_batch_size

        # Concatenate all features
        features = np.concatenate(features_list, axis=0)

        return features

    def calculate_fid(
        self,
        real_dataloader,
        generator,
        num_samples=50000,
        batch_size=50,
        latent_dim=100,
    ):
        """
        Calculate FID score between real and generated images.

        Args:
            real_dataloader: DataLoader providing real images
            generator: Generator model
            num_samples: Number of samples to use
            batch_size: Batch size for generation
            latent_dim: Dimensionality of latent space

        Returns:
            fid_score: Fréchet Inception Distance
        """
        # Extract features from real images
        real_features = self.extract_features(real_dataloader, num_samples)

        # Extract features from generated images
        fake_features = self.extract_features_from_generator(
            generator, num_samples, batch_size, latent_dim
        )

        # Calculate FID score
        fid_score = calculate_fid(real_features, fake_features)

        return fid_score
