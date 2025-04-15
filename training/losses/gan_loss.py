import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss:
    """
    Standard GAN loss functions.
    Based on the original GAN paper by Goodfellow et al.
    """
    def __init__(self, loss_type='vanilla'):
        """
        Initialize the GAN loss.
        
        Args:
            loss_type: Type of GAN loss ('vanilla', 'lsgan', 'wgan', 'hinge')
        """
        self.loss_type = loss_type
        
        # Binary cross entropy loss for vanilla GAN
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
    
    def generator_loss(self, fake_pred):
        """
        Calculate the generator loss.
        
        Args:
            fake_pred: Discriminator predictions on fake samples
            
        Returns:
            loss: Generator loss
        """
        if self.loss_type == 'vanilla':
            # Minimize log(1 - D(G(z))) by maximizing D(G(z))
            # which is equivalent to minimizing -log(D(G(z)))
            target_real = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target_real)
        
        elif self.loss_type == 'lsgan':
            # Least squares GAN loss for generator
            target_real = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target_real)
        
        elif self.loss_type == 'wgan':
            # WGAN loss for generator
            # Maximize E[D(G(z))]
            return -torch.mean(fake_pred)
        
        elif self.loss_type == 'hinge':
            # Hinge loss for generator
            return -torch.mean(fake_pred)
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Calculate the discriminator loss.
        
        Args:
            real_pred: Discriminator predictions on real samples
            fake_pred: Discriminator predictions on fake samples
            
        Returns:
            loss: Discriminator loss
        """
        if self.loss_type == 'vanilla':
            # Maximize log(D(x)) + log(1 - D(G(z)))
            target_real = torch.ones_like(real_pred)
            target_fake = torch.zeros_like(fake_pred)
            loss_real = self.criterion(real_pred, target_real)
            loss_fake = self.criterion(fake_pred, target_fake)
            return loss_real + loss_fake
        
        elif self.loss_type == 'lsgan':
            # Least squares GAN loss for discriminator
            target_real = torch.ones_like(real_pred)
            target_fake = torch.zeros_like(fake_pred)
            loss_real = self.criterion(real_pred, target_real)
            loss_fake = self.criterion(fake_pred, target_fake)
            return loss_real + loss_fake
        
        elif self.loss_type == 'wgan':
            # WGAN loss for discriminator
            # Maximize E[D(x)] - E[D(G(z))]
            return -torch.mean(real_pred) + torch.mean(fake_pred)
        
        elif self.loss_type == 'hinge':
            # Hinge loss for discriminator
            loss_real = torch.mean(F.relu(1 - real_pred))
            loss_fake = torch.mean(F.relu(1 + fake_pred))
            return loss_real + loss_fake
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


# Wasserstein loss with gradient penalty
class WassersteinGPLoss:
    """
    Wasserstein GAN loss with gradient penalty.
    Based on the WGAN-GP paper.
    """
    def __init__(self, lambda_gp=10.0):
        """
        Initialize the WGAN-GP loss.
        
        Args:
            lambda_gp: Gradient penalty coefficient
        """
        self.lambda_gp = lambda_gp
    
    def generator_loss(self, fake_pred):
        """
        Calculate the generator loss.
        
        Args:
            fake_pred: Critic predictions on fake samples
            
        Returns:
            loss: Generator loss
        """
        # Maximize E[D(G(z))]
        return -torch.mean(fake_pred)
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Calculate the critic loss (without gradient penalty).
        
        Args:
            real_pred: Critic predictions on real samples
            fake_pred: Critic predictions on fake samples
            
        Returns:
            loss: Critic loss
        """
        # Maximize E[D(x)] - E[D(G(z))]
        return -torch.mean(real_pred) + torch.mean(fake_pred)
    
    def gradient_penalty(self, discriminator, real_samples, fake_samples, device):
        """
        Calculate the gradient penalty.
        
        Args:
            discriminator: Critic model
            real_samples: Real data samples
            fake_samples: Generated samples
            device: Device to run the computation on
            
        Returns:
            gradient_penalty: Gradient penalty term
        """
        # Random weight for interpolation
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        
        # Interpolate between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Calculate critic scores on interpolated samples
        d_interpolates = discriminator(interpolates)
        
        # Create fake gradients
        fake = torch.ones(d_interpolates.size()).to(device)
        
        # Calculate gradients of critic scores with respect to interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty