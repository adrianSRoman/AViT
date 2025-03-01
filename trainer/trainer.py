import torch
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from trainer.base_trainer import BaseTrainer
plt.switch_backend('agg')

import wandb

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (feats, label) in enumerate(self.train_data_loader):
            label = label.to(self.device, dtype=torch.float32, non_blocking=True)
            feats = feats.to(self.device, dtype=torch.float32, non_blocking=True)
            self.optimizer.zero_grad()
            pred = self.model(feats)
            loss = self.loss_function(pred, label)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        print("Loss train", loss_total / dl_len)
        wandb.log({"Loss/train": loss_total / dl_len}, step=epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0

        for i, (feats, label) in enumerate(self.validation_data_loader):
            label = label.to(self.device, dtype=torch.float32, non_blocking=True)
            feats = feats.to(self.device, dtype=torch.float32, non_blocking=True)
            self.optimizer.zero_grad()
            pred = self.model(feats)
            loss = self.loss_function(pred, label)

            loss_total += loss.item()

        dl_len = len(self.validation_data_loader)
        val_loss_avg = loss_total / dl_len
        print("Loss validation", val_loss_avg)
        wandb.log({"Loss/val": val_loss_avg}, step=epoch)
        
        return val_loss_avg
