"""Loss functions module initialization"""

from .base_gan_loss import BaseGANLoss
from .vanilla_gan_loss import VanillaGANLoss
from .wgan_loss import WGANLoss
from .wgan_gp_loss import WGANGPLoss
from .prog_gan_loss import ProgGANGPLoss

__all__ = [
    "BaseGANLoss",
    "VanillaGANLoss",
    "WGANLoss",
    "WGANGPLoss",
	"ProgGANGPLoss"
]
