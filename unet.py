import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import cv2


def visualize(**images):
    """Plot images in one row."""
    overlay_color = [
        0,
        255,
        0,
    ]  # This is the color for the overlay, change it as per your needs. Here it's red.

    n = len(images)
    plt.figure(figsize=(10, 6))  # Adjust figure size to accommodate more images
    index = 1
    for name, image in images.items():
        image = np.squeeze(image)  # remove singleton dimensions
        plt.subplot(n, 3, index)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if "mask" in name:  # if the image is a mask
            plt.imshow(image, cmap="gray")  # use a grayscale colormap
        else:
            plt.imshow(image)
        index += 1

    # Create a colored image to overlay
    pred_mask = np.squeeze(
        images["pred_mask"]
    )  # Assuming the predicted mask is passed with key 'pred_mask'
    overlay = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    overlay[pred_mask > 0] = overlay_color

    # Merge the overlay with the original image
    image_to_show = np.squeeze(
        images["image"]
    )  # Assuming the original image is passed with key 'image'
    image_to_show = cv2.normalize(
        image_to_show, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )  # Ensure same data type

    overlayed_image = cv2.addWeighted(
        image_to_show, 1, overlay, 0.5, 0
    )  # Change the alpha values as needed

    # Overlayed Image
    plt.subplot(n, 3, index)
    plt.xticks([])
    plt.yticks([])
    plt.title(" ".join("pred_mask".split("_")).title() + " Overlay")
    plt.imshow(overlayed_image)
    plt.show()


def calculate_class_weights(mask_dir):
    class_0_count = 0
    class_1_count = 0
    mask_files = os.listdir(mask_dir)

    for mask_file in mask_files:
        mask = Image.open(os.path.join(mask_dir, mask_file))
        mask = np.array(mask)
        mask = mask / 255
        class_1_count += np.sum(mask)
        class_0_count += mask.size - np.sum(mask)

    total_count = class_0_count + class_1_count
    class_0_weight = total_count / class_0_count
    class_1_weight = total_count / class_1_count

    return class_0_weight, class_1_weight


def dice_score(pred, target, smooth=1):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )

    return dice.mean()


def dice_loss(pred, target, smooth=1):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )

    return 1 - dice


transform = transforms.Compose(
    [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class HistopathologyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

        # Verify that each image has a corresponding mask.
        self.img_names = [
            img_name
            for img_name in self.img_names
            if os.path.exists(
                os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")
            )
        ]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.splitext(self.img_names[idx])[
            0
        ]  # get the filename without extension
        img_path = os.path.join(
            self.img_dir, img_name + ".jpg"
        )  # append the image file extension
        mask_path = os.path.join(
            self.mask_dir, img_name + ".png"
        )  # append the mask file extension

        img_file = Image.open(img_path)
        mask_file = Image.open(mask_path)

        image = img_file.convert("RGB")

        mask = np.array(mask_file, dtype=np.uint8)
        mask = mask / 255

        mask_file.close()

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        image = np.array(image)
        img_file.close()

        mask = np.array(mask)
        return image, mask


class Unet(pl.LightningModule):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, class_weights=None):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        if class_weights is not None:
            self.class_weights = (
                torch.tensor(class_weights).float().to(torch.device("mps"))
            )
        else:
            self.class_weights = None

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2), double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=False):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    self.up = nn.ConvTranspose2d(
                        in_channels // 2, in_channels // 2, kernel_size=2, stride=2
                    )

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )
                x = torch.cat([x2, x1], dim=1)  ## why 1?
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def calculate_dice_and_bce_loss(self, y_hat, y, class_weights=None):
        # Apply sigmoid and threshold to get the masks
        y_hat_sigmoid = torch.sigmoid(y_hat)
        y_hat_mask = (y_hat_sigmoid > 0.5).float()
        # Calculate BCE Loss
        if class_weights is not None:
            weights = torch.ones_like(y) * class_weights[0]
            weights[y == 1] = class_weights[1]
            bce_loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=weights)
            loss = bce_loss + dice_loss(y_hat_mask, y).mean()
        else:
            bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
            loss = bce_loss  # + dice_loss(y_hat_mask, y).mean()

        # Calculate Dice score
        dice = dice_score(y_hat_mask, y)

        return loss, dice

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss, dice = self.calculate_dice_and_bce_loss(
            y_hat, y, class_weights=self.class_weights
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice_score", dice, prog_bar=True)
        return {"loss": loss, "dice": dice}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss, dice = self.calculate_dice_and_bce_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice_score", dice, prog_bar=True)
        return {"loss": loss, "dice": dice}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss, dice = self.calculate_dice_and_bce_loss(y_hat, y)
        y_hat_sigmoid = torch.sigmoid(y_hat)
        pred_mask = (y_hat_sigmoid > 0.5).float()
        visualize(
            image=x[0].permute(1, 2, 0).cpu().numpy(),
            mask=y[0].cpu().numpy(),
            pred_mask=pred_mask[0].cpu().numpy(),
            pred_mask_raw=y_hat_sigmoid[0].cpu().detach().numpy(),
        )
        self.log("test_loss", loss)
        self.log("test_dice_score", dice, prog_bar=True)
        return {"loss": loss, "dice": dice}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [scheduler]


def main(args):
    # Load datasets
    if args["class_weights"]:
        print("Calculating class weights...")
        class_0_weight, class_1_weight = calculate_class_weights(args["train_masks"])
        print(f"Class 0 weight: {class_0_weight}, Class 1 weight: {class_1_weight}")

    print("Loading datasets...")
    train_dataset = HistopathologyDataset(
        args["train_images"], args["train_masks"], transform=transform
    )
    val_dataset = HistopathologyDataset(
        args["val_images"], args["val_masks"], transform=test_transform
    )
    test_dataset = HistopathologyDataset(
        args["test_images"], args["test_masks"], transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=False,
    )

    if args.get("test_only"):
        # Load model from checkpoint
        print("Loading model from checkpoint...")
        model = Unet.load_from_checkpoint(args["checkpoint_path"])

        # Initialize a trainer
        trainer = pl.Trainer(devices="auto")

        # Test the model
        print("Testing the model...")
        trainer.test(model, dataloaders=test_loader)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=("checkpoints"),
            verbose=True,
            monitor="val_loss",
            mode="min",
            save_last=True,
            filename="model-{epoch:02d}-{val_loss:.2f}",
        )
        stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
        )

        # Initialize the U-Net model
        if args.get("continue_from_checkpoint"):
            print("Continuing training from checkpoint...")
            model = Unet.load_from_checkpoint(args["checkpoint_path"])
        elif args.get("class_weights"):
            model = Unet(class_weights=[class_0_weight, class_1_weight])
        else:
            model = Unet()

        # Initialize a trainer
        trainer = pl.Trainer(
            max_epochs=args["epochs"],
            devices="auto",
            fast_dev_run=args["debug"],
            callbacks=[checkpoint_callback, stop_callback],
        )

        # Train the model
        print("Training the model...")
        trainer.fit(model, train_loader, val_loader)
        print("Training complete!")
        print("Testing the model...")
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a U-Net model for histopathology image segmentation"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
