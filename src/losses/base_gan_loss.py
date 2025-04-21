"""
Base GAN Loss Module

This module provides the abstract base class for implementing various GAN loss functions.
All custom GAN loss implementations should inherit from this base class and implement its abstract methods.
"""

from abc import ABC, abstractmethod
import torch


class BaseGANLoss(ABC):
    """
    Abstract base class for GAN loss functions.
    
    This class defines the interface for implementing different GAN loss variants.
    All GAN loss implementations must inherit from this class and implement
    the generator_loss and discriminator_loss methods.
    """

    @abstractmethod
    def generator_loss(self, fake_pred: torch.Tensor, fake_label: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the generator.
        
        Args:
            fake_pred (torch.Tensor): Discriminator predictions on generated (fake) data
            
        Returns:
            torch.Tensor: Computed loss value for the generator
        """

    @abstractmethod
    def discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor,
        real_label: torch.Tensor, fake_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss for the discriminator.
        
        Args:
            real_pred (torch.Tensor): Discriminator predictions on real data
            fake_pred (torch.Tensor): Discriminator predictions on generated (fake) data
            
        Returns:
            torch.Tensor: Computed loss value for the discriminator
        """
