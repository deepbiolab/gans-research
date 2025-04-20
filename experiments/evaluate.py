"""
Comprehensive evaluation script for GAN models.
Evaluates quality (FID), coverage (Precision-Recall), and generation speed.
"""

import os
import time
import argparse
import logging
import yaml
import numpy as np
import torch

from experiments.inference import load_model
from src.data.dataloader import create_dataloader
from src.utils.feature_extraction import extract_features
from src.metrics.quality import calculate_fid
from src.metrics.coverage import compute_precision_recall
from src.utils.set_experiment import configure_experiment, setup_logger, setup_summary
from src.utils.visualization import visualize_feature_space, visualize_metrics


def evaluate_quality(real_features, fake_features):
    """
    Evaluate image quality using FID score.

    Args:
        logger: Logger instance

    Returns:
        fid_score: Fréchet Inception Distance score
    """

    # Calculate FID
    fid_score = calculate_fid(real_features, fake_features)

    return fid_score


def evaluate_coverage(real_features, fake_features):
    """
    Evaluate coverage using Precision and Recall metrics.

    Args:
        real_features: Features of real images
        fake_features: Features of generated images

    Returns:
        precision: Precision metric
        recall: Recall metric
        f1_score: F1 Score metric
    """

    # Calculate precision and recall
    precision, recall = compute_precision_recall(
        real_features, fake_features, nearest_k=5
    )

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1_score


def evaluate_speed(model, config):
    """
    Evaluate image generation speed.

    Args:
        model: GAN model to evaluate
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        batch_times: Dictionary mapping batch sizes to (avg_time, std_time)
        best_time: Best generation time per image in milliseconds
    """
    # Get parameters from config
    batch_sizes = [1, 16, 64, 256]  # Test different batch sizes
    num_repeats = 10  # Number of repeats for reliable timing
    latent_dim = config["model"]["latent_dim"]
    device = torch.device(config["experiment"]["device"])

    # Ensure model is on the correct device
    model.to(device)
    model.eval()

    # Warm up
    with torch.no_grad():
        z = torch.randn(10, latent_dim).to(device)
        _ = model.generate_images(latent_vectors=z)

    # Measure generation time for different batch sizes
    batch_times = {}

    for batch_size in batch_sizes:
        times = []
        for _ in range(num_repeats):
            z = torch.randn(batch_size, latent_dim).to(device)

            # Measure time
            start_time = time.time()
            with torch.no_grad():
                _ = model.generate_images(latent_vectors=z)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()

            end_time = time.time()

            # Calculate time per image
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            time_per_image = elapsed_time / batch_size
            times.append(time_per_image)

        # Calculate average and standard deviation
        avg_time = np.mean(times)
        std_time = np.std(times)
        batch_times[batch_size] = (avg_time, std_time)

    # Get the best time (usually with largest batch size, but not always)
    best_batch_size = min(batch_times.keys(), key=lambda x: batch_times[x][0])
    best_time, best_std = batch_times[best_batch_size]

    return batch_times, best_time, best_std


def main():
    """Main evaluation function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GAN Model Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Override checkpoint path"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="GAN model name (vanilla_gan, dcgan, wgan, etc.)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Metrics to evaluate: 'quality', 'coverage', 'speed', or 'all'",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--viz_method",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="Method for feature space visualization",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = configure_experiment(config, gpu_id=args.gpu, seed=args.seed)
    config["logging"]["wandb_task"] = "Evaluate"

    # Override config with command-line options if provided
    if args.checkpoint is not None:
        config["inference"]["checkpoint_path"] = args.checkpoint
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.num_samples is not None:
        config["evaluation"]["num_samples"] = args.num_samples
    if args.batch_size is not None:
        config["evaluation"]["batch_size"] = args.batch_size

    # Create output directory
    output_dir = config["experiment"]["output_dir"]
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        output_dir=eval_dir,
        log_file="evaluation.log",
        level=logging.INFO,
        format_str="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Setup summary writer for logging visualizations
    writer = setup_summary(config, eval_dir)

    # Log configuration
    logger.info(f"Evaluation metrics: {args.metrics}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Using device: {config['experiment']['device']}")
    logger.info(f"Model name: {config['model']['name']}")
    logger.info(f"Checkpoint: {config['inference']['checkpoint_path']}")
    logger.info(f"Number of samples: {config['evaluation']['num_samples']}")
    logger.info(f"Batch size: {config['evaluation']['batch_size']}")

    # Load model
    model = load_model(config, logger)

    # Load dataset
    _, valid_dataloader = create_dataloader(config)

    # Extract features for real images and fake images
    real_features, fake_features = extract_features(model, valid_dataloader, config)

    # Initialize metrics dictionary
    metrics = {}

    # Evaluate quality (FID)
    if args.metrics in ["all", "quality"]:
        logger.info("Evaluating quality (FID)...")
        fid_score = evaluate_quality(real_features, fake_features)
        metrics["FID"] = fid_score

    # Evaluate coverage (Precision-Recall)
    if args.metrics in ["all", "coverage"]:
        logger.info("Evaluating coverage (Precision-Recall)...")
        precision, recall, f1_score = evaluate_coverage(real_features, fake_features)
        metrics["Precision"] = precision
        metrics["Recall"] = recall
        metrics["F1_Score"] = f1_score

    # Evaluate speed
    if args.metrics in ["all", "speed"]:
        logger.info("Evaluating generation speed...")
        _, best_time, _ = evaluate_speed(model, config)
        metrics["Generation_Speed_ms"] = best_time

    # Visualize results
    visualize_feature_space(
        real_features,
        fake_features,
        eval_dir,
        method=args.viz_method,
        writer=writer,
    )

    # Visualize metrics
    visualize_metrics(metrics, eval_dir, writer=writer)

    # Logging metrics
    logger.info("Summary of metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    # Close
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
