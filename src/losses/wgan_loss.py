"""
Wasserstein GAN Loss Functions

This module implements the Wasserstein GAN (WGAN) loss function, which uses
the Wasserstein-1 (Earth Mover) distance as an alternative to the traditional
GAN loss. This formulation provides more stable training and better gradient behavior.
"""

import torch
from .base_gan_loss import BaseGANLoss


class WGANLoss(BaseGANLoss):
    """
    Wasserstein GAN Loss

    Implements the WGAN loss function which estimates the Wasserstein distance
    between real and generated data distributions. Unlike traditional GAN loss,
    this formulation provides a meaningful loss metric that correlates with
    generated sample quality and doesn't suffer from vanishing gradients.

    Note: This implementation assumes the critic (discriminator) output is unbounded
    and should be used with gradient penalty or weight clipping for Lipschitz constraint.
    """

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the generator in WGAN.

        The generator tries to maximize the critic's output on generated samples,
        effectively minimizing the Wasserstein distance between real and generated
        distributions.

        Args:
            fake_pred (torch.Tensor): Critic's predictions on generated (fake) data

        Returns:
            torch.Tensor: Computed WGAN loss for the generator (negative mean of critic outputs)
        """
        # Generator wants critic to output larger values
        return -fake_pred.mean()

    def discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss for the critic (discriminator) in WGAN.

        The critic tries to maximize the difference between its outputs on real
        and fake samples, effectively estimating the Wasserstein-1 distance
        between the real and generated distributions.

        Args:
            real_pred (torch.Tensor): Critic's predictions on real data
            fake_pred (torch.Tensor): Critic's predictions on generated (fake) data

        Returns:
            torch.Tensor: Computed WGAN loss for the critic
                         (mean of fake predictions minus mean of real predictions)
        """
        # Critic maximizes the difference between real and fake predictions
        return fake_pred.mean() - real_pred.mean()
