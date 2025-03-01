import os

import torch
import librosa
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
        dataset_list = [line.rstrip('\n').split(" ") for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.labels_path = labels_path

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.labels_hop_len_s = 0.1
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode
        # TODO: Parametrize the variables below
        self.dataset_type = "foa"
        self.fs = 16000
        self.hop_len_s = 0.02
        self.hop_length = int(self.fs * self.hop_len_s) 
        self.frame_step = int(self.fs * self.labels_hop_len_s)
        self.win_len = 2 * self.hop_length
        self.n_fft = self.next_pow2(self.win_len)
        self.nb_mel_bins = 64
        self.label_seq_len = 50 # 5 seconds sequence length
        self.num_feat_chans = 10
        self.feat_seq_len = self.label_seq_len * int(self.labels_hop_len_s // self.hop_len_s)
        self.normalize_audio = False
        # audio segment length (number of samples needed for a feat sequence length)
        self.audio_segment_len = self.fs // 10 * self.label_seq_len 
        self.feats = FeatureClass(self.fs, self.n_fft, self.hop_length, self.win_len, self.nb_mel_bins)
        self.length = len(dataset_list)
        # Calculate total number of sequences
        self.length_total = sum(self.calculate_sequences_per_file(f[0]) for f in dataset_list)


    @staticmethod
    def next_pow2(x):
        return 2 ** (x - 1).bit_length()

    def get_wavefile_length(self, file_path):
        return librosa.get_duration(filename=file_path, sr=self.fs) * self.fs

    def calculate_sequences_per_file(self, file_path):
        wave_length = self.get_wavefile_length(file_path)
        return int(wave_length // self.frame_step)

    # TODO: fix the nb_classes parameter here by reading it from the json config
    def load_adpit_labels(self, label_filepath, total_label_frames, n_classes=1, seq_len=50):
        """Loads adpit labels"""
        desc_file_polar = load_output_format_file(label_filepath)
        desc_file = convert_output_format_polar_to_cartesian(desc_file_polar)
        label_mat = get_adpit_labels_for_file(desc_file, total_label_frames, n_classes)
        # TODO: here we need to index only the labels we need
        return torch.tensor(label_mat)

 
    def load_and_preprocess_audio(self, audio_path) -> Tuple[torch.Tensor, int]:
        """Load and preprocess a single audio file."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        # Resample if necessary
        if sr != self.fs:
            resampler = torchaudio.transforms.Resample(sr, self.fs)
            waveform = resampler(waveform)
        # Ensure 4 or 32 channels
        if waveform.shape[0] not in [4, 32]:
            raise ValueError(f"Expected 4 or 32 channels, got {waveform.shape[0]}")
        if waveform.shape[0] == 32:
            waveform = waveform[[5,9,25,21],:]
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

    def split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    def collate_fn(self, batch):
        features, labels, names = zip(*batch)

        # Concatenate all features and labels into tensors
        features = torch.cat([f for f in features], dim=0)
        labels = torch.cat([l for l in labels], dim=0)

        # Split into sequences
        feature_seqs = self.split_in_seqs(features, self.feat_seq_len)
        label_seqs = self.split_in_seqs(labels, self.label_seq_len)
        
        feature_seqs = feature_seqs.permute(0, 2, 1, 3)

        return feature_seqs, label_seqs, names


    def __len__(self):
        return self.length


    def __getitem__(self, item):
        audio_path = self.dataset_list[item][0] # get .wav path
        label_path = self.dataset_list[item][1] # get .csv path

        data_name = os.path.splitext(os.path.basename(audio_path))[0]
         
        # Load and preprocess audio
        waveform, sr = self.load_and_preprocess_audio(audio_path)
        # Load labels
        total_label_frames = int(waveform.shape[1] / (self.fs * self.labels_hop_len_s))
        #total_feats_frames = int(waveform.shape[1] / self.hop_length)
        
        seld_label = self.load_adpit_labels(label_path, total_label_frames)

        #print("total audio and labels length", waveform.shape, seld_label.shape)
        if waveform.shape[1] < self.audio_segment_len:
            # Pad the waveform to the required length
            pad_len = self.audio_segment_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len), mode='constant', value=0)
            # TODO: need to modify the labels here too for audio shorter than 5 seconds
            pad_label_len = self.label_seq_len - seld_label.shape[0]
            seld_label = torch.nn.functional.pad(seld_label, (0, 0, 0, 0, 0, 0, 0, pad_label_len), mode='constant', value=0)
        else:
            # Generate valid start indices that are multiples of 100ms (i.e 100 ms of audio)
            max_start_idx = waveform.shape[1] - self.audio_segment_len
            valid_indices = torch.arange(0, max_start_idx + 1, self.frame_step)
            # Select a random valid start index
            start_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
            # Slice the waveform to get a random segment
            waveform = waveform[:, start_idx:start_idx + self.audio_segment_len]
            start_idx_lb = start_idx // self.frame_step
            seld_label = seld_label[start_idx_lb:start_idx_lb + self.label_seq_len]
        
        total_feats_frames = int(waveform.shape[1] / self.hop_length)
        seld_feats = self.feats._extract_features(waveform, total_feats_frames)
        seld_feats = seld_feats.reshape(total_feats_frames, self.num_feat_chans, self.nb_mel_bins)

        # The input of model should be fixed-length in the training.
        return seld_feats, seld_label, data_name

