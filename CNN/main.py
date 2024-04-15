import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_modules import CNNModule

from lightning.pytorch.loggers import TensorBoardLogger

from dataset import load_train_test, DataAug

import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')

import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-valid", action="store_true")
    parser.add_argument("--exp", default="Experiments")
    args = parser.parse_args()

    ## Dataset
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")

    if args.test:
        logger = TensorBoardLogger(os.path.join(args.exp), version=0)
        test_loader = DataLoader(test_dataset, batch_size=250, drop_last=False, shuffle=False, num_workers=5)
        model = CNNModule.load_from_checkpoint(os.path.join(args.exp, "best.ckpt"))
        tester = L.Trainer(devices=1, accelerator="gpu", default_root_dir=args.exp, logger=logger)
        tester.test(model, dataloaders=test_loader)
    elif args.no_valid:
        torch.manual_seed(2024)
        model = CNNModule()
        train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=5)
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.exp, "checkpoints"),
                                                save_top_k=2,
                                                mode="min",
                                                monitor="Train MAPE (%)",
                                                save_last=True)
        trainer = L.Trainer(max_epochs=60,
                            devices=1, accelerator="gpu",
                            default_root_dir=args.exp,
                            callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader)
        os.rename(checkpoint_callback.best_model_path, os.path.join(args.exp, "best.ckpt"))
    else:
        ## Model
        torch.manual_seed(2024)
        model = CNNModule()

        ## Dataloader
        seed = torch.Generator().manual_seed(2024)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - 1000, 1000], generator=seed)
        # train_dataset = DataAug(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=5)
        valid_loader = DataLoader(valid_dataset, batch_size=250, drop_last=False, shuffle=False, num_workers=5)

        ## Checkpoints
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.exp, "checkpoints"),
                                                save_top_k=2,
                                                mode="min",
                                                monitor="Validation MAPE (%)",
                                                save_last=True)

        ## Training
        trainer = L.Trainer(max_epochs=60,
                            devices=1, accelerator="gpu",
                            default_root_dir=args.exp,
                            callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, valid_loader)
        os.rename(checkpoint_callback.best_model_path, os.path.join(args.exp, "best.ckpt"))
        trainer.loggers = []

        ## Testing
        trainer.validate(model, dataloaders=valid_loader, ckpt_path=os.path.join(args.exp, "best.ckpt"))
