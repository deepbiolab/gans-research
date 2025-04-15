import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def save_image_grid(images, filepath, nrow=8, padding=2, normalize=True, value_range=(-1, 1)):
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images, (B, C, H, W)
        filepath: Path to save the grid image
        nrow: Number of images displayed in each row of the grid
        padding: Amount of padding
        normalize: If True, shift the image to the range (0, 1)
        value_range: Range of input images, needed for normalization
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Make grid and save image
    try:
        # Try with newer parameter name 'value_range' first
        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range)
    except TypeError:
        try:
            # Try with older parameter name 'range'
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, range=value_range)
        except TypeError:
            # Fall back to version without range parameter for older torchvision versions
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    
    torchvision.utils.save_image(grid, filepath)
    
    return grid

def display_image_grid(images, nrow=8, padding=2, normalize=True, value_range=(-1, 1), title=None, figsize=(10, 10)):
    """
    Display a grid of images.
    
    Args:
        images: Tensor of images, (B, C, H, W)
        nrow: Number of images displayed in each row of the grid
        padding: Amount of padding
        normalize: If True, shift the image to the range (0, 1)
        value_range: Range of input images, needed for normalization
        title: Title for the plot
        figsize: Figure size
    """
    # Make grid 
    try:
        # Try with newer parameter name 'value_range' first
        grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range)
    except TypeError:
        try:
            # Try with older parameter name 'range'
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, range=value_range)
        except TypeError:
            # Fall back to version without range parameter for older torchvision versions
            grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    
    # Convert tensor to numpy array
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().detach().numpy()
        
    # Convert from CHW to HWC format
    grid = np.transpose(grid, (1, 2, 0))
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return grid

def visualize_training_progress(generator, device, fixed_noise, output_dir, iteration, nrow=8, image_name="progress"):
    """
    Generate and save images using fixed noise vectors to visualize training progress.
    
    Args:
        generator: Generator model
        device: Device to run generation on
        fixed_noise: Fixed latent vectors for consistent visualization
        output_dir: Directory to save images
        iteration: Current iteration (used for filename)
        nrow: Number of images per row
        image_name: Base name for the saved image
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
    generator.train()
    
    # Create output directory if it doesn't exist
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Save grid of generated images
    filepath = os.path.join(samples_dir, f"{image_name}_{iteration:06d}.png")
    save_image_grid(fake_images, filepath, nrow=nrow)
    
    return fake_images

def save_interpolation_grid(generator, device, start_vector, end_vector, steps=10, output_path=None, nrow=None):
    """
    Generate and save an interpolation between two latent vectors.
    
    Args:
        generator: Generator model
        device: Device to run generation on
        start_vector: Starting latent vector
        end_vector: Ending latent vector
        steps: Number of interpolation steps
        output_path: Path to save the interpolation grid
        nrow: Number of images per row (default: all in one row)
    """
    generator.eval()
    
    # Generate intermediate vectors
    alpha_values = np.linspace(0, 1, steps)
    vectors = []
    
    for alpha in alpha_values:
        interpolated = start_vector * (1 - alpha) + end_vector * alpha
        vectors.append(interpolated)
    
    # Stack all vectors
    all_vectors = torch.cat(vectors, dim=0)
    
    # Generate images
    with torch.no_grad():
        images = generator(all_vectors)
    
    # Set default number of rows if not provided
    if nrow is None:
        nrow = steps
    
    # Save or return the grid
    if output_path is not None:
        grid = save_image_grid(images, output_path, nrow=nrow)
        return grid
    else:
        grid = display_image_grid(images, nrow=nrow, title="Latent Space Interpolation")
        return grid, images