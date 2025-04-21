"""
WGAN Critic (Discriminator) module.

This class inherits the DCGAN discriminator architecture, with the
assumption that the final activation is linear (no sigmoid).
"""

from src.models.dcgan.discriminator import DCGANDiscriminator

class WGANDiscriminator(DCGANDiscriminator):
    """
    WGAN Critic.

    Inherits the DCGAN discriminator structure.
    Ensure the output layer is linear (no sigmoid).
    """
