"""
WGAN Generator module.

This class currently inherits the DCGAN generator architecture,
as recommended by the original WGAN paper. Future modifications
(e.g., for WGAN-GP or architecture ablation) can be implemented here.
"""

from src.models.dcgan.generator import DCGANGenerator

class WGANGPGenerator(DCGANGenerator):
    """
    WGAN-GP Generator.

    Inherits the DCGAN generator structure.
    """
    # Inherit everything from DCGANGenerator
