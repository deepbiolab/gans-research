# GANs Research Reproduction

```mermaid
graph LR
    subgraph Basic_Models["Basic Models (2014-2015)"]
        VanillaGAN[Vanilla GAN 2014] --> DCGAN[DCGAN 2015]
    end
    
    subgraph Training_Stability["Training Stability Path"]
        DCGAN --> WGAN[WGAN 2017]
        WGAN --> WGAN_GP[WGAN-GP 2017]
    end
    
    subgraph Control_Mechanisms["Control Mechanisms"]
        VanillaGAN --> CGAN[Conditional GAN 2014]
        note1[Explicit Control]
    end
    
    subgraph Quality_Improvement["Quality Improvement Path"]
        DCGAN --> ProgGAN[Progressive GAN 2017]
        WGAN --> StyleGAN1[StyleGAN 2018]
        ProgGAN --> StyleGAN1
        WGAN_GP -.-> StyleGAN1
        StyleGAN1 --> StyleGAN2[StyleGAN2 2019]
        StyleGAN2 --> StyleGAN3[StyleGAN3 2021]
        note2[Implicit Style Control]
    end
    
    %% Relationship Notes
    Training_Stability -. "Stability Focus" .-> Quality_Improvement
    Control_Mechanisms -. "Control Evolution" .-> Quality_Improvement
    
    %% Styling
    classDef done fill:#A9B7C0,stroke:#555,stroke-width:2px;
    classDef pending fill:#D7C4BB,stroke:#555,stroke-width:2px;
    classDef current fill:#B8C4B8,stroke:#555,stroke-width:2px;
    classDef note fill:#fff,stroke:#555,stroke-width:1px,stroke-dasharray: 5 5;
    
    %% Apply styles
    class VanillaGAN done;
    class DCGAN,WGAN,WGAN_GP,CGAN,ProgGAN,StyleGAN1,StyleGAN2,StyleGAN3 pending;
    class note1,note2 note;
    
    %% Subgraph styling
    style Basic_Models fill:#f5f5f5,stroke:#666,stroke-width:2px;
    style Training_Stability fill:#e8f4f8,stroke:#666,stroke-width:2px;
    style Control_Mechanisms fill:#f8f4e8,stroke:#666,stroke-width:2px;
    style Quality_Improvement fill:#f0f8f0,stroke:#666,stroke-width:2px;

```

This repository contains implementations of key Generative Adversarial Network (GAN) architectures, organized in an incremental learning approach.


## Implementation Progress

1. **Basic GANs**
   - [x] **Vanilla GAN**: Original formulation with fully connected layers for both generator and discriminator networks.
   - [ ] **DCGAN**: Introduced convolutional architectures and established key architectural guidelines for stable GAN training.
   
2. **Training Stability Improvements**
   - [ ] **WGAN**: Improved training stability using Wasserstein distance.
   - [ ] **WGAN-GP**: Enhanced WGAN with gradient penalty.
   
3. **Conditional Control**
   - [ ] **Conditional GAN**: Added ability to control generation with explicit labels.
   
4. **High-Quality Generation**
   - [ ] **Progressive GAN**: Introduced progressive growing methodology for generating high-resolution images.
   - [ ] **StyleGAN v1**: Combined progressive growing with style-based generation.
   - [ ] **StyleGAN v2**: Improved architecture and removing progressive growing.
   - [ ] **StyleGAN v3**: Added alias-free generation techniques.



## Installation

### Option 1: Install dependencies only
```bash
pip install -r requirements.txt
```

### Option 2: Install as a package (recommended for development)
```bash
# If you have a previous installation, remove it first
pip uninstall gans-research

# Install in editable mode
pip install -e .
```



## Usage

### Training
Each GAN implementation has its own training script in the `experiments` directory:

```bash
# Example: Train a basic GAN on MNIST
python experiments/stage1_basic/train_vanilla_gan.py
```

### Inference

This project supports flexible GAN inference with dynamic architecture selection and robust logging.  
You can easily generate sample images from any registered GAN model using a unified interface.

**Command-Line Inference**

To generate images using a trained GAN model:

```
python experiments/inference.py \
  --config path/to/your_config.yaml \
  --model_name dcgan \
  --checkpoint outputs/dcgan/checkpoints/final_model.pth \
  --num_samples 32 \
  --out outputs/dcgan/results/sample_grid.png
```

**Arguments:**

- `--config`: Path to your YAML configuration file. (**required**)
- `--model_name`: Name of the GAN model to use (e.g., `vanilla_gan`, `dcgan`, `wgan`). Overrides the config if specified.
- `--checkpoint`: Path to the trained model checkpoint. Overrides the config if specified.
- `--num_samples`: Number of images to generate. Overrides the config if specified.
- `--out`: Output path for the generated image grid. If not set, a default path will be used.
- `--gpu`: GPU ID to use (`-1` for CPU).

**Configuration Example**

Your YAML config should include the following fields:

```yaml
inference:
  model_name: vanilla_gan      # or dcgan, wgan, etc.
  checkpoint_path: outputs/vanilla_gan/checkpoints/final_model.pth
  num_samples: 16
experiment:
  output_dir: outputs/vanilla_gan
```


**Adding New Models**

To register a new GAN model, add it to `MODEL_REGISTRY` in `src/models/__init__.py`:

```python
from .my_custom_gan import MyCustomGAN

MODEL_REGISTRY = {
    "vanilla_gan": VanillaGAN,
    "dcgan": DCGAN,
    "wgan": WGAN,
    "my_custom_gan": MyCustomGAN,
}
```


**Note:**  
Generated images will be saved as a grid (default: `inference_result.png`), with grayscale images automatically converted to RGB for visualization.



## Project Structure

- `models/`: GAN implementations
- `data/`: Dataset loading and processing
- `training/`: Training utilities and loss functions
- `evaluation/`: Evaluation metrics and visualization tools
- `utils/`: Helper utilities
- `experiments/`: Training scripts for each GAN variant
- `configs/`: Configuration files



## References

- [Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661)
- [DCGAN (2015)](https://arxiv.org/abs/1511.06434)
- [Wasserstein GAN (2017)](https://arxiv.org/abs/1701.07875)
- [WGAN-GP (2017)](https://arxiv.org/abs/1704.00028)
- [Conditional GANs (2014)](https://arxiv.org/abs/1411.1784)
- [Progressive GAN (2017)](https://arxiv.org/abs/1710.10196)
- [StyleGAN (2018)](https://arxiv.org/abs/1812.04948)
- [StyleGAN2 (2019)](https://arxiv.org/abs/1912.04958)
- [StyleGAN3 (2021)](https://arxiv.org/abs/2106.12423)