import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_modules import CNNModule

from dataset import load_train_test

import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')

import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    ## Dataset
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")

    if args.test:
        test_loader = DataLoader(test_dataset, batch_size=250, drop_last=False, shuffle=False, num_workers=5)
        model = CNNModule.load_from_checkpoint("Experiments/best.ckpt", logger=False)
        tester = L.Trainer(devices=1, accelerator="gpu", logger=False)
        tester.test(model, dataloaders=test_loader)
    else:
        ## Model
        torch.manual_seed(2024)
        model = CNNModule()

        ## Dataloader
        seed = torch.Generator().manual_seed(2024)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - 1000, 1000], generator=seed)
        train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=5)
        valid_loader = DataLoader(valid_dataset, batch_size=250, drop_last=False, shuffle=False, num_workers=5)

        ## Checkpoints
        checkpoint_callback = ModelCheckpoint(dirpath="Experiments/checkpoints",
                                                save_top_k=2,
                                                mode="min",
                                                monitor="Validation MAPE (%)",
                                                save_last=True)

        ## Training
        trainer = L.Trainer(max_epochs=30,
                            devices=1, accelerator="gpu",
                            default_root_dir="Experiments",
                            callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, valid_loader)
        os.rename(checkpoint_callback.best_model_path, "Experiments/best.ckpt")
        trainer.loggers = []

        ## Testing
        trainer.validate(model, dataloaders=valid_loader, ckpt_path='Experiments/best.ckpt')
