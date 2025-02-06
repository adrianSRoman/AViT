import os

import torch
import torchaudio
from torch.utils import data

from pathlib import Path
from typing import List, Tuple, Optional


class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 limit=None,
                 offset=0,
                 sample_length=256,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <data_1_path>
            <data_2_path>
            ...

        Return:
            (multi-channel phase stft, one-hot DoA labels)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode
        # TODO: Parametrize the variables below
        self.n_fft = 256
        self.hop_length = 256
        self.sample_rate = 24000
        self.normalize_audio = False

    def load_and_preprocess_audio(self, audio_path) -> Tuple[torch.Tensor, int]:
        """Load and preprocess a single audio file."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure 4 channels
        if waveform.shape[0] != 4:
            raise ValueError(f"Expected 4 channels, got {waveform.shape[0]}")
        
        # Normalize if requested
        if self.normalize_audio:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform, self.sample_rate
    
    def compute_stft_features(self, waveform: torch.Tensor, start_idx: int) -> torch.Tensor:
        """Compute STFT features for a segment of audio."""
        # Extract segment
        segment = waveform[:, start_idx:start_idx + self.sample_length]
        # Compute STFT
        stft = torch.stft(
            segment,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True
        )
        # Convert to phase spectrogram
        phs_spec = torch.angle(stft)
        # Shape: (4, freq_bins, time_frames)
        return phs_spec

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        audio_path = self.dataset_list[item]
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        label = filename.split("_")[-1]
        
        # One-hot encode the label (dimension 4)
        one_hot_label = torch.zeros(4)
        label = int(label)
        one_hot_label[label] = 1

        # Load and preprocess audio
        waveform, _ = self.load_and_preprocess_audio(audio_path)

        # Compute STFT features
        max_start = waveform.shape[1] - self.sample_length
        start_idx = torch.randint(0, max(1, max_start), (1,)).item()
        stft_features = self.compute_stft_features(waveform, start_idx)

        # The input of model should be fixed-length in the training.
        return stft_features[:, :, 0], label

