#!/usr/bin/env python3

import lightning as L
import torch
import torchmetrics
from einops import repeat, rearrange
from torchinfo import summary
from torch import nn
from torchvision.models import squeezenet1_1


class ModelBaseClass(L.LightningModule):
    """Example model architecture for the FIOT datasets"""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch, batch_idx):
        x, y_true = batch
        y_hat = self(x)
        return self.criterion(y_hat, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def on_test_start(self):
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=self.num_classes, task="multiclass").to(self.device)
        self.test_f1 = torchmetrics.F1Score(num_classes=self.num_classes, average="macro", task="multiclass").to(self.device)

    def test_step(self, batch, batch_idx):
        """similar to step, but we also update confusion matrix and F1 score"""
        x, y_true = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.test_confmat.update(preds, y_true)
        self.test_f1.update(preds, y_true)
        loss = self.criterion(y_hat, y_true)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        self.confmat = self.test_confmat.compute()
        self.log(
            "test_acc",
            torch.sum(torch.diagonal(self.confmat)) / torch.sum(self.confmat).item(),
        )
        self.log("test_f1", self.test_f1.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        return optimizer


class HighwayBaselineModel(ModelBaseClass):
    """Example model architecture for the FIOT datasets"""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        # submodel selection
        self.submodel = squeezenet1_1(num_classes=num_classes)

    def forward(self, x):
        # add channel dimension and repeat to 3 channels in one step
        x = repeat(x, "batch height width -> batch 3 height width")
        x = self.submodel(x)
        return x


class StudentModel(ModelBaseClass):
    """Unfinished student model template"""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        # create layers (unfinished)
        self.layers = nn.Identity()

    def forward(self, x):
        # x will be of shape (batchsize, 512, 243)
        x = rearrange(x, "batchsize height width -> batchsize 1 height width")  # add channel dimension
        # x is now of shape (batchsize, 1, 512, 243)
        x = self.layers(x)
        # x should be of shape (batchsize, num_classes)
        return x


if __name__ == "__main__":
    model = HighwayBaselineModel()
    summary(model, input_data=torch.randn(1, 512, 243), device="cpu")
