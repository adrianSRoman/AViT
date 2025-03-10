# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class PositionalEmbedding(nn.Module):  # Not used in the baseline
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SeldModel(torch.nn.Module):
    def __init__(self, params, feat_config):
        super().__init__()
        t_pool_size = params["t_pool_size"]
        f_pool_size = params["f_pool_size"]
        self.conv_block_list = nn.ModuleList()
        if len(f_pool_size):
            for conv_cnt in range(len(f_pool_size)):
                self.conv_block_list.append(ConvBlock(in_channels=params["nb_cnn2d_filt"] if conv_cnt else feat_config["num_feat_chans"], out_channels=params["nb_cnn2d_filt"]))
                self.conv_block_list.append(nn.MaxPool2d((t_pool_size[conv_cnt], f_pool_size[conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params["dropout_rate"]))

        self.gru_input_dim = params["nb_cnn2d_filt"] * int(np.floor(feat_config["nb_mel_bins"] / np.prod(f_pool_size)))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params["rnn_size"],
                                num_layers=params["nb_rnn_layers"], batch_first=True,
                                dropout=params["dropout_rate"], bidirectional=True)

        # self.pos_embedder = PositionalEmbedding(self.params['params["rnn_size"]'])

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params["nb_self_attn_layers"]):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=params["rnn_size"], num_heads=params["nb_heads"], dropout=params["dropout_rate"],  batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(params["rnn_size"]))

        self.fnn_list = torch.nn.ModuleList()
        if params["nb_fnn_layers"]:
            for fc_cnt in range(params["nb_fnn_layers"]):
                self.fnn_list.append(nn.Linear(params["fnn_size"] if fc_cnt else params["rnn_size"], params["fnn_size"], bias=True))
        self.fnn_list.append(nn.Linear(params["fnn_size"] if params["nb_fnn_layers"] else params["rnn_size"], feat_config["unique_classes"]*3*3, bias=True))

    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        # pos_embedding = self.pos_embedder(x)
        # x = x + pos_embedding
        
        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x 
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa
