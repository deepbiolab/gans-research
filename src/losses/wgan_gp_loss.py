"""
Wasserstein GAN with Gradient Penalty Loss Functions

This module implements the Wasserstein GAN loss with gradient penalty (WGAN-GP),
which improves upon the original WGAN by replacing weight clipping with a gradient
penalty term to enforce the Lipschitz constraint. This results in more stable training
and better quality results.
"""

import torch
from .wgan_loss import WGANLoss

class WGANGPLoss(WGANLoss):
    """
    WGAN with Gradient Penalty Loss
    
    Extends the Wasserstein GAN loss by adding a gradient penalty term that enforces
    the Lipschitz constraint on the critic (discriminator). This approach provides
    better training stability compared to weight clipping used in the original WGAN.
    
    The gradient penalty ensures that the gradient norm of the critic's output with
    respect to its input is close to 1 (1-Lipschitz constraint) at interpolated points
    between real and fake samples.
    """

    def gradient_penalty(
        self,
        discriminator,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """
        Calculate the gradient penalty term for WGAN-GP.
        
        Computes the gradient penalty by interpolating between real and fake samples
        and ensuring that the gradient norm at these points is close to 1. This enforces
        the Lipschitz constraint on the critic.
        
        Args:
            discriminator: The critic model that takes input data and outputs predictions
            real_data (torch.Tensor): Batch of real data samples
            fake_data (torch.Tensor): Batch of generated (fake) data samples
            lambda_gp (float, optional): Gradient penalty coefficient. Defaults to 10.0
            
        Returns:
            torch.Tensor: Computed gradient penalty term (λ * E[(||∇D(x̂)||₂ - 1)²])
            
        Note:
            The interpolation is done using random epsilon values for each sample
            in the batch: x̂ = ε * x_real + (1 - ε) * x_fake
        """
        batch_size = real_data.size(0)
        # Generate random interpolation points
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        
        # Create interpolated samples between real and fake data
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates.requires_grad_(True)
        
        # Get critic predictions for interpolated samples
        d_interpolates = discriminator(interpolates)
        grad_outputs = torch.ones_like(d_interpolates)

        # Calculate gradients of critic output with respect to inputs
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        
        # Compute gradient penalty: (||∇D(x̂)||₂ - 1)²
        gp = ((grad_norm - 1) ** 2).mean()
        return lambda_gp * gp