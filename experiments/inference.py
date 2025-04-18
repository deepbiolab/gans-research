"""This provides a common interface for different generator architectures to generate sample images."""

import os
import argparse
import logging
import yaml

from src.models import MODEL_REGISTRY
from src.utils.set_experiment import configure_experiment, setup_logger
from src.utils.visualization import make_grid, save_grid


def load_model(config, logger):
    """
    Load a pre-trained GAN model based on the configuration.
    """
    model_name = config["inference"].get("model_name", "vanilla_gan")
    checkpoint_path = config["inference"]["checkpoint_path"]

    if model_name not in MODEL_REGISTRY:
        logger.error(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
        raise ValueError(f"Unknown model: {model_name}")

    gan_model = MODEL_REGISTRY[model_name]
    logger.info(f"Using model: {model_name} ({gan_model.__name__})")
    model = gan_model(config)
    model.load(checkpoint_path)
    model.eval()
    return model


def infer_and_save(model, config, logger, output_path=None, num_samples=None):
    """
    Generate images using the model and save them to a file.
    """
    output_dir = config["experiment"]["output_dir"]
    result_dir = os.path.join(output_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = output_path or os.path.join(result_dir, "inference_result.png")

    nsamples = num_samples or config["inference"]["num_samples"]
    logger.info(f"Generating {nsamples} samples...")
    images = model.generate_images(batch_size=nsamples)

    # Convert grayscale to RGB if needed
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    grid = make_grid(images, nrow=8)
    save_grid(grid, result_path)
    logger.info(f"Saved inference result to {result_path}")


def main():
    """Main function to load a GAN model, generate images, and save them."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GAN Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for CPU")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Override number of samples"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Override checkpoint path"
    )
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="GAN model name (vanilla_gan, dcgan, wgan, etc.)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = configure_experiment(config, gpu_id=args.gpu)

    # Override config with command-line options if provided
    if args.checkpoint is not None:
        config["inference"]["checkpoint_path"] = args.checkpoint
    if args.num_samples is not None:
        config["inference"]["num_samples"] = args.num_samples
    if args.model_name is not None:
        config["inference"]["model_name"] = args.model_name

    # Setup logger (logs to both file and console)
    output_dir = config["experiment"]["output_dir"]
    logger = setup_logger(
        output_dir=output_dir,
        log_file="inference.log",
        level=logging.INFO,
        format_str="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {config['inference']['checkpoint_path']}")
    logger.info(f"Num samples: {config['inference']['num_samples']}")
    logger.info(f"Model name: {config['inference'].get('model_name', 'vanilla_gan')}")

    model = load_model(config, logger)
    infer_and_save(
        model, config, logger, output_path=args.out, num_samples=args.num_samples
    )
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
