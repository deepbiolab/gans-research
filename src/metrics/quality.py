"""
Utilities for calculating quality of generated images
by using Fréchet Inception Distance (FID) between real and generated images.
"""

import numpy as np
from scipy import linalg
import torch

from src.utils.feature_extraction import (
    InceptionV3Features,
    extract_features_for_real_images,
    extract_features_for_fake_images,
)


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
        self.device = device if device is not None else torch.device("cpu")
        self.feature_extractor = InceptionV3Features().to(self.device)
        self.feature_extractor.eval()

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
        real_features = extract_features_for_real_images(
            feature_extractor=self.feature_extractor,
            dataloader=real_dataloader,
            device=self.device,
            num_samples=num_samples,
        )

        # Extract features from generated images
        fake_features = extract_features_for_fake_images(
            feature_extractor=self.feature_extractor,
            generator=generator,
            device=self.device,
            num_samples=num_samples,
            batch_size=batch_size,
            latent_dim=latent_dim,
        )

        # Calculate FID score
        fid_score = calculate_fid(real_features, fake_features)

        return fid_score
