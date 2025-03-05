import os
import torch
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from trainer.base_trainer import BaseTrainer
from util.utils import get_multi_accdoa_labels, write_output_format_file, reshape_3Dto2D, determine_similar_location 
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
            test_dataloader
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.test_data_loader = test_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (feats, label, data_name) in enumerate(self.train_data_loader):
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

        for i, (feats, label, data_name) in enumerate(self.validation_data_loader):
            label = label.to(self.device, dtype=torch.float32, non_blocking=True)
            feats = feats.to(self.device, dtype=torch.float32, non_blocking=True)
            self.optimizer.zero_grad()
            pred = self.model(feats)
            loss = self.loss_function(pred, label)

            loss_total += loss.item()
            
        #########################################
        ####### Compute Validation Loss #########
        #########################################
        dl_len = len(self.validation_data_loader)
        val_loss_avg = loss_total / dl_len
        print("Loss validation", val_loss_avg)
        wandb.log({"Loss/val": val_loss_avg}, step=epoch)

        return val_loss_avg


    @torch.no_grad()
    def _test_epoch(self):

        for i, (feats, label, data_name) in enumerate(self.test_data_loader):
            label = label.to(self.device, dtype=torch.float32, non_blocking=True)
            feats = feats.to(self.device, dtype=torch.float32, non_blocking=True)
            self.optimizer.zero_grad()
            pred = self.model(feats)

            ########################################
            ####### Compute Multi-ACCDOA Preds #####
            ########################################
            sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(pred.detach().cpu().numpy(), 1)
            sed_pred0 = reshape_3Dto2D(sed_pred0)
            doa_pred0 = reshape_3Dto2D(doa_pred0)
            sed_pred1 = reshape_3Dto2D(sed_pred1)
            doa_pred1 = reshape_3Dto2D(doa_pred1)
            sed_pred2 = reshape_3Dto2D(sed_pred2)
            doa_pred2 = reshape_3Dto2D(doa_pred2)

            # dump SELD results to the correspondin file
            output_file = os.path.join("./output", f"{data_name[0]}.csv")
            output_dict = {}
            for frame_cnt in range(sed_pred0.shape[0]):
                for class_cnt in range(sed_pred0.shape[1]):
                    # determine whether track0 is similar to track1
                    flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, 15, 1)
                    flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, 15, 1)
                    flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, 15, 1)
                    # unify or not unify according to flag
                    if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                        if sed_pred0[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+1], doa_pred0[frame_cnt][class_cnt+2*1]])
                        if sed_pred1[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+1], doa_pred1[frame_cnt][class_cnt+2*1]])
                        if sed_pred2[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+1], doa_pred2[frame_cnt][class_cnt+2*1]])
                    elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        if flag_0sim1:
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+1], doa_pred2[frame_cnt][class_cnt+2*1]])
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+1], doa_pred_fc[class_cnt+2*1]])
                        elif flag_1sim2:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+1], doa_pred0[frame_cnt][class_cnt+2*1]])
                            doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+1], doa_pred_fc[class_cnt+2*1]])
                        elif flag_2sim0:
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+1], doa_pred1[frame_cnt][class_cnt+2*1]])
                            doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+1], doa_pred_fc[class_cnt+2*1]])
                    elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                        output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+1], doa_pred_fc[class_cnt+2*1]])

            write_output_format_file(output_file, output_dict)
