import lightning as L
import os
import cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from facades_dataset import FacadesDataset
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient

import torchvision
from mmodel import *


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image


def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f"{folder_name}/epoch_{epoch}", exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f"{folder_name}/epoch_{epoch}/result_{i + 1}.png", comparison)


class LitModel(L.LightningModule):
    def __init__(self, model, adversarial):
        super().__init__()
        self.model = model
        self.adversarial = adversarial
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

    # def criterion(self, pred, target):
    #     return F.binary_cross_entropy(pred, target)

    def forward(self, image_rgb):
        return self.model(image_rgb)

    def training_step(self, batch, batch_idx):
        rgb, seg = batch
        B = rgb.shape[0]
        fake_label = torch.zeros(B, 1).to(self.device)
        real_label = torch.ones(B, 1).to(self.device)
        o_model, o_adversarial = self.optimizers()
        pred = self.model(rgb)

        # Train the model
        START_ADV = 5
        self.toggle_optimizer(o_model)
        loss_g = self.criterion(self.adversarial(rgb, pred), real_label)
        loss_g_direct = F.mse_loss(pred, seg)
        self.log("loss_g", loss_g)
        self.log("loss_g_l1", loss_g_direct)
        self.manual_backward(
            (loss_g + loss_g_direct * 100)
            if self.current_epoch > START_ADV
            else loss_g_direct
        )
        o_model.step()
        o_model.zero_grad()
        self.untoggle_optimizer(o_model)

        # Train the discriminator
        if self.current_epoch > START_ADV:
            self.toggle_optimizer(o_adversarial)
            fake = self.adversarial(rgb, pred.detach())
            real = self.adversarial(rgb, seg)
            loss_d = (
                self.criterion(fake, fake_label) + self.criterion(real, real_label)
            ) * 0.5
            self.log("loss_d", loss_d)
            self.manual_backward(loss_d)
            o_adversarial.step()
            o_adversarial.zero_grad()
            self.untoggle_optimizer(o_adversarial)

        if self.current_epoch % 10 == 0 and batch_idx == 0:
            # save_images(inputs, targets, outputs, 'train_results', self.current_epoch)
            b = min(B, 4)
            mlf_logger: MlflowClient = self.logger.experiment
            grid = tensor_to_image(
                torchvision.utils.make_grid(
                    torch.cat([rgb[:b], seg[:b], pred[:b]], dim=0), nrow=b
                )
            )
            run = self.logger.run_id
            os.makedirs("train_results", exist_ok=True)
            file_name = f"train_results/train_epoch_{self.current_epoch}.png"
            cv2.imwrite(file_name, grid)
            # Now, grid is a numpy array of shape (H, W, C) with values in [0, 255] and dtype uint8
            mlf_logger.log_artifact(run, file_name)

    def validation_step(self, batch, batch_idx):
        rgb, seg = batch
        B = rgb.shape[0]
        pred = self.model(rgb)
        loss = F.mse_loss(pred, seg)
        self.log("val_L1", loss)

        if batch_idx == 0:
            # save_images(inputs, targets, outputs, 'train_results', self.current_epoch)
            b = min(B, 4)
            mlf_logger: MlflowClient = self.logger.experiment
            grid = tensor_to_image(
                torchvision.utils.make_grid(
                    torch.cat([rgb[:b], seg[:b], pred[:b]], dim=0), nrow=b
                )
            )
            run = self.logger.run_id
            os.makedirs("train_results", exist_ok=True)
            file_name = f"train_results/val_epoch_{self.current_epoch}.png"
            cv2.imwrite(file_name, grid)
            # Now, grid is a numpy array of shape (H, W, C) with values in [0, 255] and dtype uint8
            mlf_logger.log_artifact(run, file_name)

    def configure_optimizers(self):
        lr = 2e-4
        b1 = 0.5
        b2 = 0.999
        o1 = optim.AdamW(
            self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=1e-3
        )
        o2 = optim.AdamW(
            self.adversarial.parameters(), lr=lr, betas=(b1, b2), weight_decay=1e-3
        )
        return [
            {
                "optimizer": o1,
            },
            {
                "optimizer": o2,
            },
        ]


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = FacadesDataset(list_file="train_list.txt")
        self.val = FacadesDataset(list_file="val_list.txt")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val, batch_size=self.batch_size * 8, shuffle=False
        )


def weight_init(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
    else:
        print(f"No reset_parameters in {type(m)}")


if __name__ == "__main__":
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    model = UNet()
    adversarial = Discriminator()
    lit_model = LitModel(model, adversarial)
    lit_model.apply(weight_init)
    trainer = L.Trainer(
        log_every_n_steps=1,
        max_epochs=10000,
        check_val_every_n_epoch=3,
        logger=MLFlowLogger(experiment_name="facades", log_model=True),
    )
    trainer.fit(lit_model, datamodule=DataModule())
