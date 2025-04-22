"""This provides a common interface for different generator architectures to generate sample images."""

import os
import argparse
import logging
import yaml
import torch
from src.models import AutoModel
from src.utils.set_experiment import configure_experiment, setup_logger
from src.utils.visualization import make_grid, save_grid


def load_model(config, logger):
    """
    Load a pre-trained GAN model based on the configuration.
    """
    model_name = config["model"].get("name", "vanilla_gan")
    checkpoint_path = config["inference"]["checkpoint_path"]

    device = config["experiment"]["device"]

    try:
        model = AutoModel.from_pretrained(
            model_name=model_name,
            path=checkpoint_path,
            map_location=device,
            config=config,
        )
        logger.info(f"Successfully loaded {model.__class__.__name__} from {checkpoint_path}")
    except ValueError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    model.eval()
    return model


def generate(model, config, logger, num_samples=None, labels=None):
    """
    Generate images using the model.

    Args:
        model: The GAN model
        config: Configuration dictionary
        logger: Logger instance
        num_samples: Number of samples per label (if labels specified), otherwise total samples
        labels: List of class labels for conditional generation

    Returns:
        Generated images tensor
    """
    model_name = config["model"].get("name", "").lower()
    is_conditional = "cgan" in model_name

    if is_conditional and labels is not None:
        labels = torch.tensor(labels, device=config["experiment"]["device"])
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        nlabels = labels.shape[0]
        samples_per_label = num_samples or 1
        logger.info(f"Generating {samples_per_label} samples for each label in {labels.tolist()}...")
        all_images = []
        all_labels = []
        for label in labels:
            label_batch = label.repeat(samples_per_label)
            images = model.generate_images(batch_size=samples_per_label, labels=label_batch)
            all_images.append(images)
            all_labels.extend([label.item()] * samples_per_label)
        images = torch.cat(all_images, dim=0)
        logger.info(f"Total generated images: {images.shape[0]}")
    else:
        nsamples = num_samples or config["inference"]["num_samples"]
        logger.info(f"Generating {nsamples} samples...")
        images = model.generate_images(batch_size=nsamples)

    # Convert grayscale to RGB if needed
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    return images


def save_images(images, config, logger, output_path=None, nrow=10):
    """
    Save generated images to file.

    Args:
        images: Generated image tensor
        config: Configuration dictionary
        logger: Logger instance
        output_path: Override path to save images
        nrow: Number of images per row in the grid

    Returns:
        Path to the saved image file
    """
    output_dir = config["experiment"]["output_dir"]
    result_dir = os.path.join(output_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = output_path or os.path.join(result_dir, "inference_result.png")

    grid = make_grid(images, nrow=nrow)
    save_grid(grid, result_path)
    logger.info(f"Saved inference result to {result_path}")

    return result_path


def main():
    """Main function to load a GAN model, generate images, and save them."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GAN Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Override number of samples"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated list of class labels for conditional generation (e.g. 0,1,2,3)",
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
    config = configure_experiment(config, gpu_id=args.gpu, seed=args.seed)

    labels = None
    if args.labels is not None:
        labels = [int(x) for x in args.labels.split(",")]

    # Override config with command-line options if provided
    if args.checkpoint is not None:
        config["inference"]["checkpoint_path"] = args.checkpoint
    if args.num_samples is not None:
        config["inference"]["num_samples"] = args.num_samples
    if args.model_name is not None:
        config["model"]["name"] = args.model_name

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
    logger.info(f"Model name: {config['model']['name']}")

    model = load_model(config, logger)

    # Generate images
    images = generate(model, config, logger, num_samples=args.num_samples, labels=labels)

    # Save images
    save_images(images, config, logger, output_path=args.out)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
