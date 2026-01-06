#!/usr/bin/env python3
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from segmentation_models_pytorch.encoders import mobileone
from torchinfo import summary

from radiomana import *
from radiomana.datasets import HighwayDataModule


def train():
    # select model
    ModelClass = NanoGRU

    shortname = ModelClass.__name__.lower()

    # setup model & data
    dmodule = HighwayDataModule(num_workers=16, batch_size=256)
    model = ModelClass()
    _ = summary(model, input_data=torch.randn(1, 512, 243), device="cpu", depth=4)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename=shortname + "-{epoch:03d}-{val_loss:05f}",
            save_top_k=1,
            mode="min",
        ),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=50, mode="min"),
    ]

    # do training
    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(accelerator="gpu", devices=2, max_epochs=-1, precision=32, callbacks=callbacks)
    trainer.fit(model, datamodule=dmodule)

    # rewind to best checkpoint and test
    print(f"rewinding to best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    best_model = ModelClass.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    results = trainer.test(best_model, datamodule=dmodule)[0]

    # print per-class accuracy
    confmat = best_model.confmat
    print("confusion matrix:\n", confmat)
    class_acc = torch.diagonal(confmat) / confmat.sum(dim=1)
    print("test per-class accuracy:")
    for cdx, class_label in enumerate(dmodule.data_test.class_labels):
        print(f"  class {cdx} ({class_label:<22s}): {class_acc[cdx]:8.3%}")

    # save best model weights
    torch.save(best_model.state_dict(), f"{shortname}-loss={results['test_loss']:0.3f}.pt")

    print(f"test loss: {results['test_loss']:0.3f}")
    print(f"test f1: {results['test_f1']:0.3f}")
    print(f"test acc: {results['test_acc']:8.3%}")

    # load best model weights (disabled, as we already have best_model)
    # best_model = HighwayBaselineModel()
    # best_model.load_state_dict(torch.load("example.pt"))


if __name__ == "__main__":
    train()
