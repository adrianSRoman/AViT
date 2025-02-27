import math
import torch
import librosa
import torchaudio
import itertools

import numpy as np

class FeatureClass:
    def __init__(self, fs, nfft, hop_len, win_len, nb_mel_bins):
        self._fs = fs
        self.nfft = nfft
        self.hop_len = hop_len
        self.win_len = win_len
        self.nb_mel_bins = nb_mel_bins
        self.window = torch.hann_window(win_len)
        self.mel_wts = torch.tensor(librosa.filters.mel(sr=self._fs, n_fft=self.nfft, n_mels=self.nb_mel_bins).T, dtype=torch.float32)

    def _spectrogram(self, audio_input, _nb_frames):
        """
        Compute the spectrogram using PyTorch Audio.

        Args:
            audio_input (torch.Tensor): Input audio tensor of shape (time, channels).
            _nb_frames (int): Number of frames to limit the output spectrogram.

        Returns:
            torch.Tensor: Spectrogram of shape (channels, time, frequency).
        """
        audio_input = torch.transpose(audio_input, 1, 0)
        _nb_ch = audio_input.shape[1]
        spectra = []

        for ch_cnt in range(_nb_ch):
            stft_ch = torch.stft(
                input=audio_input[:, ch_cnt],
                n_fft=self.nfft,
                hop_length=self.hop_len,
                win_length=self.win_len,
                window=self.window,
                return_complex=True
            )
            # Only keep the first `_nb_frames`
            stft_ch = stft_ch[:, :_nb_frames]
            spectra.append(stft_ch)
        return torch.stack(spectra).T # time, frequency, channels

    def _get_mel_spectrogram(self, linear_spectra):
        """
        Convert linear spectrogram to mel spectrogram using PyTorch.

        Args:
            linear_spectra: Complex STFT [frames, frequency_bins, channels]
        
        Returns:
            Mel spectrogram features [frames, channels*mel_bins]
        """
        # Initialize output tensor
        batch_size = linear_spectra.shape[0]
        nb_channels = linear_spectra.shape[-1]
        mel_feat = torch.zeros((batch_size, self.nb_mel_bins, nb_channels), device=linear_spectra.device)
    
        for ch_cnt in range(nb_channels):
            # Calculate magnitude spectrogram
            mag_spectra = torch.abs(linear_spectra[:, :, ch_cnt])**2
        
            # Apply mel filterbank
            mel_spectra = torch.matmul(mag_spectra, self.mel_wts)
        
            # Convert to dB scale (librosa.power_to_db equivalent)
            log_mel_spectra = 10.0 * torch.log10(torch.clamp(mel_spectra, min=1e-10))
        
            mel_feat[:, :, ch_cnt] = log_mel_spectra
    
        # Reshape to [frames, channels*mel_bins]
        mel_feat = mel_feat.transpose(1, 2).reshape(batch_size, -1)
    
        return mel_feat


    def _get_gcc(self, linear_spectra):
        """
        Extract generalized cross-correlation (GCC) features using PyTorch.
    
        Args:
            linear_spectra: Complex STFT [frames, frequency_bins, channels]
        
        Returns:
            GCC features [frames, channels_combinations*mel_bins]
        """
        batch_size = linear_spectra.shape[0]
        nb_channels = linear_spectra.shape[-1]
    
        # Calculate number of channel combinations
        def nCr(n, r):
            import math
            return math.factorial(n) // (math.factorial(r) * math.factorial(n-r))
    
        gcc_channels = nCr(nb_channels, 2)
    
        # Initialize output tensor
        gcc_feat = torch.zeros((batch_size, self.nb_mel_bins, gcc_channels), device=linear_spectra.device)
    
        cnt = 0
        for m in range(nb_channels):
            for n in range(m+1, nb_channels):
                # Calculate cross-spectrum
                R = torch.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
            
                # Normalize and compute inverse FFT
                cc = torch.fft.irfft(torch.exp(1j * torch.angle(R)))
            
                # Rearrange to center the features
                half_bins = self.nb_mel_bins // 2
                cc = torch.cat([cc[:, -half_bins:], cc[:, :half_bins]], dim=-1)
            
                gcc_feat[:, :, cnt] = cc
                cnt += 1
    
        # Reshape to [frames, channels_combinations*mel_bins]
        return gcc_feat.transpose(1, 2).reshape(batch_size, -1)

    def _extract_features(self, audio_in, nb_feat_frames):
        spec = self._spectrogram(audio_in, nb_feat_frames)
        mel_spect = self._get_mel_spectrogram(spec)
        # extract gcc
        gcc = self._get_gcc(spec)
        feat = torch.concatenate((mel_spect, gcc), axis=-1)
        return feat


