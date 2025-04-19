"""
This module contains the registry of all the models.
"""

from .vanilla_gan import VanillaGAN

MODEL_REGISTRY = {
    "vanilla_gan": VanillaGAN,
}
