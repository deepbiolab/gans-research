"""
Vanilla GAN Loss Functions

This module implements the standard (vanilla) GAN loss function using binary cross-entropy.
It provides the traditional non-saturating GAN loss formulation as described in the original GAN paper.
"""

import torch
import torch.nn.functional as F
from .base_gan_loss import BaseGANLoss

class VanillaGANLoss(BaseGANLoss):
    """
    Standard GAN Loss (non-saturating, cross-entropy)
    
    Implements the original GAN loss function using binary cross-entropy loss.
    This is the non-saturating version where the generator maximizes log(D(G(z)))
    instead of minimizing log(1 - D(G(z))).
    """

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate the non-saturating loss for the generator.
        
        The generator tries to maximize the probability of the discriminator
        predicting its generated samples as real (label 1).
        
        Args:
            fake_pred (torch.Tensor): Discriminator predictions on generated (fake) data
            
        Returns:
            torch.Tensor: Computed binary cross-entropy loss for the generator
        """
        # Generator wants discriminator to output 1
        return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

    def discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss for the discriminator.
        
        The discriminator tries to maximize the probability of correctly classifying
        real samples as real (label 1) and fake samples as fake (label 0).
        
        Args:
            real_pred (torch.Tensor): Discriminator predictions on real data
            fake_pred (torch.Tensor): Discriminator predictions on generated (fake) data
            
        Returns:
            torch.Tensor: Computed average binary cross-entropy loss for the discriminator
                         (average of real and fake sample losses)
        """
        # Discriminator wants real samples to be 1 and fake samples to be 0
        loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        return (loss_real + loss_fake) / 2