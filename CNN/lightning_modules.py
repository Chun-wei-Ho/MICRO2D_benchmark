import torch
from torchmetrics.regression import MeanAbsolutePercentageError

import lightning as L
from model import BasicCNN

class CNNModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BasicCNN()
        self.loss = MeanAbsolutePercentageError()
    def training_step(self, batch, batch_idx, mode="Train"):
        x, y_ref = batch
        y_hyp = self.model(x)
        loss = self.loss(y_hyp, y_ref)
        self.log(f"{mode} MAPE (%)", loss * 100)
        return loss
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, mode="Validation")
    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, mode="Test")
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer