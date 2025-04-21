"""
Visualization utilities for qualitative assessment of models.
"""

import os
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.manifold import TSNE

import torch
import torchvision


def make_grid(
    samples: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple = (-1, 1),
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        samples: Tensor of samples, (B, C, H, W)
        nrow: Number of images displayed in each row of the grid
        padding: Amount of padding
        normalize: If True, shift the image to the range (0, 1)
        value_range: Range of input images, needed for normalization
    Returns:
        grid: Grid of images, (C, H * nrow, W * ncol)
    """
    grid = torchvision.utils.make_grid(
        samples,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )
    return grid


def save_grid(grid: torch.Tensor, filepath: str) -> None:
    """Save a grid of images."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert tensor to numpy array
    grid = grid.cpu().detach().numpy()

    # Convert from CHW to HWC format
    grid = np.transpose(grid, (1, 2, 0))

    # Convert to PIL image for resizing
    if grid.shape[2] == 1:  # Grayscale
        grid = Image.fromarray((grid[:, :, 0] * 255).astype(np.uint8), mode="L")
    else:  # RGB
        grid = Image.fromarray((grid * 255).astype(np.uint8))

    # Resize grid
    width, height = grid.size
    new_width = int(width * 2.0)
    new_height = int(height * 2.0)
    grid_resized = grid.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized grid
    grid_resized.save(filepath, dpi=(300, 300), optimize=True)


def display_grid(grid: torch.Tensor, title: str = None, figsize: tuple = (10, 10)):
    """Display a grid of images."""

    # Convert tensor to numpy array
    grid = grid.cpu().detach().numpy()

    # Convert from CHW to HWC format
    grid = np.transpose(grid, (1, 2, 0))

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(grid)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return grid


def extract_number(filename):
    """Extract the iteration number from a filename."""
    match = re.search(r"(\d+)\.png$", filename)
    if match:
        return int(match.group(1))
    return 0


def create_animation(
    experiment_dir: str,
    samples_subdir: str = "samples",
    pattern: str = "progress_*.png",
    duration: int = 200,
    include_iteration_text: bool = True,
) -> str:
    """
    Create a animation from training progress images.

    Args:
        experiment_dir: Root directory of the experiment
        samples_subdir: Subdirectory containing sample images
        pattern: Glob pattern to match image files
        duration: Duration of each frame in milliseconds (for GIF)
        include_iteration_text: Whether to add iteration number as text overlay

    Returns:
        Path to the created animation file
    """

    # Construct paths
    samples_dir = os.path.join(experiment_dir, samples_subdir)
    output_path = os.path.join(experiment_dir, "training_animation.gif")

    # Find all matching image files
    image_files = glob.glob(os.path.join(samples_dir, pattern))

    if not image_files:
        raise ValueError(
            f"No images found matching pattern '{pattern}' in {samples_dir}"
        )

    # Sort the image files by iteration number
    image_files.sort(key=extract_number)

    # Load images
    images = []
    for file in image_files:
        img = Image.open(file)

        # Add iteration text if requested
        if include_iteration_text:
            iteration = extract_number(file)
            draw = ImageDraw.Draw(img)

            # Try to get a font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
            except IOError:
                font = ImageFont.load_default()

            # Add text at the bottom of the image
            text = f"Iteration: {iteration}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            position = ((img.width - text_width) // 2, img.height - text_height - 10)

            # Draw text with black outline for better visibility
            for offset in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                draw.text(
                    (position[0] + offset[0], position[1] + offset[1]),
                    text,
                    fill="black",
                    font=font,
                )
            draw.text(position, text, fill="white", font=font)

        images.append(img)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
        quality=90,
    )

    return output_path


def visualize_feature_space(
    real_features, fake_features, eval_dir, method="tsne", writer=None
):
    """
    Visualize real and generated samples in feature space using dimensionality reduction.

    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        method: Dimensionality reduction method ('tsne' or 'umap')
        logger: Logger instance
        writer: Summary writer for logging visualizations

    Returns:
        fig: Figure object with the visualization
    """
    # Subsample features if there are too many
    max_samples = 1000
    if len(real_features) > max_samples:
        indices = np.random.choice(len(real_features), max_samples, replace=False)
        real_features_sub = real_features[indices]
    else:
        real_features_sub = real_features

    if len(fake_features) > max_samples:
        indices = np.random.choice(len(fake_features), max_samples, replace=False)
        fake_features_sub = fake_features[indices]
    else:
        fake_features_sub = fake_features

    # Combine features
    combined_features = np.vstack([real_features_sub, fake_features_sub])

    # Create labels
    real_labels = np.ones(len(real_features_sub))
    fake_labels = np.zeros(len(fake_features_sub))
    labels = np.hstack([real_labels, fake_labels])

    # Apply dimensionality reduction
    if method == "umap":
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(combined_features)
    else:
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        embedding = tsne.fit_transform(combined_features)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.7,
        s=5,
    )

    # Add legend and title
    method_name = "UMAP" if method == "umap" else "t-SNE"
    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=["Generated", "Real"],
        title="Data Type",
    )
    ax.add_artist(legend)
    ax.set_title(f"Feature Space Visualization using {method_name}")
    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")

    # Log to writer if available
    if writer is not None:
        writer.add_image(f"feature_space_{method_name.lower()}", fig, 0)

    # Save figure
    fig.savefig(os.path.join(eval_dir, f"feature_space_{method}.png"))
    plt.close(fig)


def visualize_metrics(metrics, eval_dir, writer=None):
    """
    Visualize GAN evaluation metrics in grouped bar charts.

    Groups:
        - Quality: [FID]
        - Coverage: [Precision, Recall, F1_Score]
        - Speed: [Generation_Speed_ms]
    """
    # Define metric groups
    quality_metrics = ["FID"]
    coverage_metrics = ["Precision", "Recall", "F1_Score"]
    speed_metrics = ["Generation_Speed_ms"]

    # Prepare grouped data
    groups = [
        ("Quality", quality_metrics),
        ("Coverage", coverage_metrics),
        ("Speed", speed_metrics),
    ]

    # Set up subplots: one for each group
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 6))

    if len(groups) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, (group_name, group_keys) in zip(axes, groups):
        # Extract values for this group; handle missing keys gracefully
        values = [metrics.get(k, 0.0) for k in group_keys]
        bars = ax.bar(group_keys, values, color="skyblue")

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * (max(values) if max(values) > 0 else 1),
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Set labels and title for each subplot
        ax.set_ylabel("Score")
        ax.set_title(f"{group_name} Metrics")
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

    fig.suptitle("GAN Evaluation Metrics (Grouped)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Log to writer if available
    if writer is not None:
        writer.add_image("summary_chart_grouped", fig, 0)

    # Save figure
    fig.savefig(os.path.join(eval_dir, "metrics_summary.png"))
    plt.close(fig)
