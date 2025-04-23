"""
Training script for ProGAN with WGAN-GP, progressive growing, and fade-in.
"""

import argparse
import yaml
import torch
from src.models import ProgGAN
from src.losses import WGANGPLoss
from src.training import GANTrainer
from src.data.dataloader import create_dataloader
from src.utils.set_experiment import configure_experiment

try:
    from ema_pytorch import EMA
except ImportError:
    EMA = None


class ProGANGPTrainer(GANTrainer):
    """
    ProGAN Trainer: WGAN-GP loss, drift penalty, EMA, progressive growing (fade-in).
    """

    def __init__(
        self,
        model,
        config,
        train_dataloader,
        valid_dataloader,
        loss_fn,
    ):
        super().__init__(model, config, train_dataloader, valid_dataloader, loss_fn)

        tricks = config

        # Drift penalty
        self.epsilon_drift = tricks.get("training", {}).get("epsilon_drift", 0.001)
        
		# n_critic
        self.n_critic = tricks.get("training", {}).get("n_critic", 1)
        
		# EMA
        self.use_ema = tricks.get("ema", {}).get("enable", False)
        self.ema = None
        if self.use_ema and EMA is not None:
            self.ema = EMA(
                self.model.generator,
                beta=tricks["ema"].get("beta", 0.999),
                update_after_step=tricks["ema"].get("update_after_step", 100),
                update_every=tricks["ema"].get("update_every", 1),
            )
            self.logger.info("EMA enabled for generator.")
        elif self.use_ema:
            self.logger.warning("EMA requested but ema_pytorch is not installed.")

        # Progressive growing
        # 需要在config["progressive"]中指定：resolutions, images_per_stage, fadein_kimgs
        self.progressive = tricks.get("progressive", {})
        self.resolutions = self.progressive.get("resolutions", [4, 8, 16, 32, 64])
        self.images_per_stage = self.progressive.get("images_per_stage", 800_000)
        self.fadein_kimgs = self.progressive.get("fadein_kimgs", 800_000)
        self.cur_stage = 0
        self.total_images = 0
        self.fadein = True
        self.alpha = 0.0
        self.max_stage = len(self.resolutions) - 1

        # 初始化分辨率
        self.set_stage(0)

    def set_stage(self, stage):
        """
        切换到指定stage，设置模型分辨率和DataLoader
        """
        self.cur_stage = stage
        res = self.resolutions[stage]
        self.model.set_resolution(res)
        # 如果DataLoader需要切换分辨率，重新创建
        self.train_dataloader, self.valid_dataloader = create_dataloader(
            self.config, override_image_size=res
        )
        self.logger.info(
            f"Switched to resolution {res}x{res}, stage {stage}/{self.max_stage}"
        )
        self.alpha = 0.0
        self.model.set_fadein_alpha(self.alpha)
        self.fadein = True

    def update_fadein(self, images_processed):
        """
        根据已处理图片数更新fade-in alpha
        """
        if self.fadein:
            fadein_images = self.fadein_kimgs
            self.alpha = min(1.0, images_processed / fadein_images)
            self.model.set_fadein_alpha(self.alpha)
            if self.alpha >= 1.0:
                self.fadein = False
                self.logger.info(
                    f"Fade-in complete for resolution {self.resolutions[self.cur_stage]}x{self.resolutions[self.cur_stage]}"
                )

    def maybe_advance_stage(self):
        """
        根据已处理图片数判断是否需要提升分辨率
        """
        images_in_stage = self.total_images - self.cur_stage * (
            self.images_per_stage + self.fadein_kimgs
        )
        if (
            images_in_stage >= self.images_per_stage + self.fadein_kimgs
            and self.cur_stage < self.max_stage
        ):
            self.set_stage(self.cur_stage + 1)

    def train_step(self, real_batch, iteration):
        """
        单步训练，支持fade-in和progressive growing
        """
        if isinstance(real_batch, (list, tuple)):
            real_imgs = real_batch[0].to(self.device)
        else:
            real_imgs = real_batch.to(self.device)
        batch_size = real_imgs.size(0)
        self.total_images += batch_size

        # 更新fade-in
        if self.fadein:
            images_in_fadein = self.total_images - (
                self.cur_stage * (self.images_per_stage + self.fadein_kimgs)
            )
            self.update_fadein(images_in_fadein)
        # 是否提升stage
        self.maybe_advance_stage()

        # 1. 训练判别器 n_critic 次
        d_loss_total = 0.0
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()
            z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            fake_imgs = self.model.generator(z).detach()
            real_validity = self.model.discriminator(real_imgs)
            fake_validity = self.model.discriminator(fake_imgs)
            gp = self.criterion.gradient_penalty(
                self.model.discriminator, real_imgs, fake_imgs, self.device
            )
            drift = self.epsilon_drift * (real_validity**2).mean()
            d_loss = (
                self.criterion.discriminator_loss(real_validity, fake_validity)
                + gp
                + drift
            )
            d_loss.backward()
            self.d_optimizer.step()
            d_loss_total += d_loss.item()
        d_loss_total /= self.n_critic

        # 2. 训练生成器
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z)
        fake_validity = self.model.discriminator(fake_imgs)
        g_loss = self.criterion.generator_loss(fake_validity)
        g_loss.backward()
        self.g_optimizer.step()

        if self.ema is not None:
            self.ema.update()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss_total,
            "gp": gp.item(),
            "drift": drift.item(),
            "alpha": self.alpha,
            "stage": self.cur_stage,
            "resolution": self.resolutions[self.cur_stage],
        }

    def generate_samples(self, sampling_num: int = 16) -> torch.Tensor:
        """
        Generate and return sample images using (optionally) EMA weights.
        """
        self.model.eval()
        generator = (
            self.ema.ema_model if (self.ema is not None) else self.model.generator
        )
        with torch.no_grad():
            z = torch.randn(sampling_num, self.model.latent_dim, device=self.device)
            samples = generator(z)
        self.model.train()
        return samples


def main():
    """
    Main entry point for ProGAN training script.
    """
    parser = argparse.ArgumentParser(
        description="Train ProGAN with WGAN-GP and progressive growing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prog_gan.yaml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure experiment environment
    config = configure_experiment(config, gpu_id=args.gpu)

    # Create dataloader (初始分辨率)
    train_dataloader, valid_dataloader = create_dataloader(config)

    # Create model
    model = ProgGAN(config)

    # Create loss function
    loss_fn = WGANGPLoss(lambda_gp=config.get("training", {}).get("lambda_gp", 10.0))

    # Create ProGANGPTrainer and train
    trainer = ProGANGPTrainer(
        model, config, train_dataloader, valid_dataloader, loss_fn=loss_fn
    )
    trainer.train()


if __name__ == "__main__":
    main()
