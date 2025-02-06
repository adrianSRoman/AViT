import os

import torch
import torchaudio
from torch.utils import data

from pathlib import Path
from typing import List, Tuple, Optional

from util.utils import load_output_format_file, convert_output_format_polar_to_cartesian, get_adpit_labels_for_file

from dataset.features import FeatureClass

class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 labels_path,
                 limit=None,
                 offset=0,
                 sample_length=12000,
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

        self.labels_path = labels_path

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.length = len(dataset_list)
        self.labels_step = 0.1
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode
        # TODO: Parametrize the variables below
        self.dataset_type = "foa"
        self.fs = 24000
        self.n_fft = 256
        self._hop_len_s = 0.02
        self.hop_length = int(self.fs * self._hop_len_s)
        self.win_len = 2 * self.hop_length
        self.n_fft = self.next_pow2(self.win_len)
        self.nb_mel_bins = 64
        self.normalize_audio = False

        self.feat_extractor = FeatureClass(self.fs, self.n_fft, self.nb_mel_bins)


    @staticmethod
    def next_pow2(x):
        return 2 ** (x - 1).bit_length()


    def load_adpit_labels(self, label_filepath, total_label_frames, n_classes=13, seq_len=50):
        """Loads adpit labels"""
        desc_file_polar = load_output_format_file(label_filepath)
        desc_file = convert_output_format_polar_to_cartesian(desc_file_polar)
        label_mat = get_adpit_labels_for_file(desc_file, total_label_frames, n_classes)
        # TODO: here we need to index only the labels we need
        return label_mat

 
    def load_and_preprocess_audio(self, audio_path) -> Tuple[torch.Tensor, int]:
        """Load and preprocess a single audio file."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.fs:
            resampler = torchaudio.transforms.Resample(sr, self.fs)
            waveform = resampler(waveform)
        
        # Ensure 4 channels
        if waveform.shape[0] != 4:
            raise ValueError(f"Expected 4 channels, got {waveform.shape[0]}")
        
        # Normalize if requested
        if self.normalize_audio:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform, sr
    
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
        data_src = os.path.basename(os.path.dirname(audio_path))

        # TODO: improve this way of handling the file name, better make the txt file contain .wav .csv
        label_path = os.path.join(self.labels_path, data_src, f"{filename}.csv")
         
        # Load and preprocess audio
        waveform, sr = self.load_and_preprocess_audio(audio_path)
        total_label_frames = int(waveform.shape[1] / (self.fs * self.labels_step))
        label = self.load_adpit_labels(label_path, total_label_frames)

        label = 0
        
        # Compute STFT features
        max_start = waveform.shape[1] - self.sample_length
        start_idx = torch.randint(0, max(1, max_start), (1,)).item()
        stft_features = self.compute_stft_features(waveform, start_idx)

        # The input of model should be fixed-length in the training.
        return stft_features[:, :, 0], label

