import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

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

        for i, (stft_feats, label) in enumerate(self.train_data_loader):
            label = label.to(self.device)
            stft_feats = stft_feats.to(self.device)
            stft_feats = stft_feats.unsqueeze(1)
            self.optimizer.zero_grad()
            pred = self.model(stft_feats)
            loss = self.loss_function(pred, label)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        wandb.log({"Loss/train": loss_total / dl_len}, step=epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0

        for i, (stft_feats, label) in enumerate(self.validation_data_loader):
            label = label.to(self.device)
            stft_feats = stft_feats.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(stft_feats)
            loss = self.loss_function(pred, label)
            # loss.backward()
            # self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        wandb.log({"Loss/val": loss_total / dl_len}, step=epoch)

