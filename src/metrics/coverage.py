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


def compute_precision_recall(real_features, fake_features, nearest_k=5):
    """
    Compute GAN precision and recall metrics between real and fake features.

    Args:
        real_features: numpy array of shape [N_real, D]
        fake_features: numpy array of shape [N_fake, D]
        nearest_k: int, k for k-nearest neighbors (typically 5)

    Returns:
        precision: float, fraction of fake samples in real manifold
        recall: float, fraction of real samples in fake manifold
    """

    # 1. Calculate the k-nearest neighbor radius for both real and fake features
    real_nn = NearestNeighbors(n_neighbors=nearest_k).fit(real_features)
    real_distances, _ = real_nn.kneighbors(real_features)
    real_radii = real_distances[:, -1]  # Manifold radius for each real point

    fake_nn = NearestNeighbors(n_neighbors=nearest_k).fit(fake_features)
    fake_distances, _ = fake_nn.kneighbors(fake_features)
    fake_radii = fake_distances[:, -1]  # Manifold radius for each fake point

    # 2. Calculate precision: fake points to real manifold
    # For each fake point, find the nearest real point and check 
    # if the distance is within the radius of that real point
    real_nn_1 = NearestNeighbors(n_neighbors=1).fit(real_features)
    fake_to_real_dist, fake_to_real_idx = real_nn_1.kneighbors(fake_features)
    fake_to_real_dist = fake_to_real_dist.flatten()
    fake_to_real_idx = fake_to_real_idx.flatten()
    # Check if the point falls within the real manifold
    precision_mask = fake_to_real_dist <= real_radii[fake_to_real_idx]
    precision = np.mean(precision_mask)

    # 3. Calculate recall: real points to fake manifold
    fake_nn_1 = NearestNeighbors(n_neighbors=1).fit(fake_features)
    real_to_fake_dist, real_to_fake_idx = fake_nn_1.kneighbors(real_features)
    real_to_fake_dist = real_to_fake_dist.flatten()
    real_to_fake_idx = real_to_fake_idx.flatten()
    # Check if the point falls within the fake manifold
    recall_mask = real_to_fake_dist <= fake_radii[real_to_fake_idx]
    recall = np.mean(recall_mask)

    return float(precision), float(recall)


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

        # Compute precision and recall
        precision, recall = compute_precision_recall(
            real_features, fake_features, nearest_k=self.nearest_k
        )

        return precision, recall, real_features, fake_features
