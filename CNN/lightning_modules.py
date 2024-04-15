import torch
from torchmetrics.regression import MeanAbsolutePercentageError

import lightning as L
from model import BasicCNN, ResNet50, DenseNet201

import matplotlib.pyplot as plt

import numpy as np

class CNNModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = BasicCNN()
        self.model = ResNet50()
        # self.model = DenseNet201()
        self.loss = MeanAbsolutePercentageError()
    def training_step(self, batch, batch_idx, mode="Train", num_pretrain=0):
        x, y_ref = batch
        y_hyp = self.model(x)
        if mode != "Train" or self.current_epoch >= num_pretrain:
            y_ref = y_ref[:, 0]
            y_hyp = y_hyp[:, 0]
        prefix = "multi objective " if y_ref.ndim == 2 else ""
        loss = self.loss(y_hyp, y_ref)
        self.log(f"{mode} {prefix}MAPE (%)", loss * 100)
        return loss
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, mode="Validation")
    def on_test_epoch_start(self):
        self.y_ref = []
        self.y_hyp = []
    def test_step(self, batch, batch_idx):
        x, y_ref = batch
        y_hyp = self.model(x)
        y_ref = y_ref[:, 0]
        y_hyp = y_hyp[:, 0]
        self.y_ref.append(y_ref)
        self.y_hyp.append(y_hyp)
    def on_test_epoch_end(self):
        y_ref = torch.cat(self.y_ref)
        y_hyp = torch.cat(self.y_hyp)
        loss = self.loss(y_hyp, y_ref)
        y_ref = y_ref.cpu().numpy()
        y_hyp = y_hyp.cpu().numpy()

        fig, ax = plt.subplots()
        ax.scatter(y_ref, y_hyp, c='b')
        xx = [y_ref.min(), y_ref.max()]; ax.plot(xx, xx, 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.grid(True)
        self.logger.experiment.add_figure("CNN Parity Plot", fig)
        self.log("Test MAPE (%)", loss * 100)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer