"""
This module contains the registry of all the models.
"""

from .vanilla_gan import VanillaGAN
from .dcgan import DCGAN
from .wgan import WGAN

MODEL_REGISTRY = {
    "vanilla_gan": VanillaGAN,
    "dcgan": DCGAN,
    "wgan": WGAN,
}


class AutoModel:
    """
    Auto class to load the appropriate GAN model based on name.

    Provides a unified interface for loading different GAN variants
    similar to HuggingFace's AutoModel pattern.
    """

    @classmethod
    def from_pretrained(cls, model_name, path, map_location=None, **kwargs):
        """
        Load a pre-trained GAN model based on model name.

        Args:
            model_name: Name of the model in MODEL_REGISTRY
            path: Path to the checkpoint file
            map_location: Device mapping for torch.load
            **kwargs: Additional arguments passed to the model's from_pretrained method

        Returns:
            A loaded model instance

        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_class = MODEL_REGISTRY[model_name]
        return model_class.from_pretrained(path, map_location=map_location, **kwargs)


__all__ = ["VanillaGAN",
           "DCGAN",
           "WGAN",
           "MODEL_REGISTRY", "AutoModel"]
