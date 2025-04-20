"""
Utilities for calculating coverage of generated images
by using Precision and Recall metrics for GAN evaluation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

from src.utils.feature_extraction import (
    InceptionV3Features,
    extract_features_for_real_images,
    extract_features_for_fake_images,
)


def compute_manifold_overlap(features_a, features_b, radii_b):
    """
    Compute manifold overlap for precision/recall calculation.

    Args:
        features_a: First set of features
        features_b: Second set of features
        radii_b: Radii for the second set of features

    Returns:
        overlap: Fraction of points in manifold overlap
    """
    # Find nearest neighbors
    nearest = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(
        features_b
    )
    distances, _ = nearest.kneighbors(features_a)
    distances = distances.flatten()

    # Count points with distance less than corresponding radius
    count = 0
    for i, dist in enumerate(distances):
        if dist <= radii_b[i]:
            count += 1

    return count / len(features_a)


def compute_precision_recall(real_features, fake_features, nearest_k=5):
    """
    Compute precision and recall metrics.

    Args:
        real_features: Features from real images
        fake_features: Features from generated images

    Returns:
        precision: Precision metric
        recall: Recall metric
    """
    # Compute manifold radii
    real_nearest = NearestNeighbors(
        n_neighbors=nearest_k, algorithm="auto", n_jobs=-1
    ).fit(real_features)
    fake_nearest = NearestNeighbors(
        n_neighbors=nearest_k, algorithm="auto", n_jobs=-1
    ).fit(fake_features)

    # For each real sample, find its nearest neighbors in the fake data
    real_to_fake_distances, _ = real_nearest.kneighbors(fake_features)
    fake_to_real_distances, _ = fake_nearest.kneighbors(real_features)

    # Use the k-th nearest neighbor to determine radius
    real_radii = np.max(real_to_fake_distances, axis=1)
    fake_radii = np.max(fake_to_real_distances, axis=1)

    # Compute precision and recall
    precision = compute_manifold_overlap(fake_features, real_features, real_radii)
    recall = compute_manifold_overlap(real_features, fake_features, fake_radii)

    return precision, recall


class PrecisionRecallCalculator:
    """
    Calculator for Precision and Recall metrics between real and generated images.
    Based on "Improved Precision and Recall Metric for Assessing Generative Models".
    """

    def __init__(self, device=None, nearest_k=5):
        """
        Initialize the precision-recall calculator.

        Args:
            feature_extractor: Model to extract features from images
            device: Device to use for computation
            k: Manifold estimation parameter
            nearest_k: Number of nearest neighbors to consider
        """
        self.device = device if device is not None else torch.device("cpu")
        self.feature_extractor = InceptionV3Features().to(self.device)
        self.feature_extractor.eval()

        self.nearest_k = nearest_k

    def calculate_metrics(
        self,
        real_dataloader,
        generator,
        num_samples=10000,
        batch_size=50,
        latent_dim=100,
    ):
        """
        Calculate precision and recall metrics.

        Args:
            real_dataloader: DataLoader providing real images
            generator: Generator model
            num_samples: Number of samples to use
            batch_size: Batch size for generation
            latent_dim: Dimensionality of latent space

        Returns:
            precision: Precision metric
            recall: Recall metric
            real_features: Feature vectors from real images
            fake_features: Feature vectors from generated images
        """
        # Extract features from real images
        real_features = extract_features_for_real_images(
            feature_extractor=self.feature_extractor,
            dataloader=real_dataloader,
            device=self.device,
            num_samples=num_samples,
            desc="Extracting real features for PR",
        )

        # Extract features from generated images
        fake_features = extract_features_for_fake_images(
            feature_extractor=self.feature_extractor,
            generator=generator,
            device=self.device,
            num_samples=num_samples,
            batch_size=batch_size,
            latent_dim=latent_dim,
            desc="Extracting generated features for PR",
        )

        # Compute precision and recall
        precision, recall = compute_precision_recall(real_features, fake_features, nearest_k=self.nearest_k)

        return precision, recall, real_features, fake_features
