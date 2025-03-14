import torch
import torch.nn as nn
import librosa
import numpy as np
import torchaudio

class LogmelFilterBank(nn.Module):
    def __init__(self, sr, n_fft, n_mels, fmin, fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True):
        """
        Convert spectrogram to log-mel spectrogram.
        
        Args:
            sr: Sample rate
            n_fft: FFT size
            n_mels: Number of mel filter banks
            fmin: Minimum frequency in mel filterbank
            fmax: Maximum frequency in mel filterbank
            ref: Reference value for computing decibels
            amin: Minimum threshold for stabilized log
            top_db: Maximum DB difference for clipping
            freeze_parameters: If True, parameters will not be updated during training
        """
        super(LogmelFilterBank, self).__init__()
        
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.freeze_parameters = freeze_parameters
        
        # Create mel filter bank matrix
        self.melW = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        
        # Convert to tensor
        self.melW = torch.tensor(self.melW, dtype=torch.float32)
        
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Spectrogram, (batch_size, n_freqs, time_steps)
        
        Returns:
            Log mel spectrogram: (batch_size, n_mels, time_steps)
        """
        # Apply mel filterbank
        mel_spectrogram = torch.matmul(x, self.melW.to(x.device))
        
        # Convert to log scale
        log_mel_spectrogram = torch.log10(torch.clamp(mel_spectrogram, min=self.amin))
        log_mel_spectrogram = 20.0 * log_mel_spectrogram - 20.0 * torch.log10(torch.tensor(self.ref))
        
        # Apply top_db clipping if specified
        if self.top_db is not None:
            max_val = torch.max(log_mel_spectrogram, dim=1, keepdim=True)[0]
            log_mel_spectrogram = torch.clamp(
                log_mel_spectrogram, min=(max_val - self.top_db).expand_as(log_mel_spectrogram)
            )
        
        return log_mel_spectrogram


class STFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=320, win_length=1024, window='hann', 
                center=True, pad_mode='reflect', freeze_parameters=True):
        """
        Short-time Fourier transform layer.
        
        Args:
            n_fft: FFT size
            hop_length: Hop size
            win_length: Window length
            window: Window type
            center: Whether to pad input on both sides
            pad_mode: Padding mode
            freeze_parameters: If True, parameters will not be updated during training
        """
        super(STFT, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        
        # Create window function
        if window == 'hann':
            self.window_tensor = torch.hann_window(win_length)
        else:
            raise ValueError(f"Window function '{window}' not supported")
        
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Audio signal, (batch_size, samples)
        
        Returns:
            Real and imaginary parts of STFT, each (batch_size, n_freqs, time_steps)
        """
        # Compute STFT
        complex_stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window_tensor.to(x.device),
            center=self.center, 
            pad_mode=self.pad_mode, 
            return_complex=True
        )
        
        # Split into real and imaginary components
        real = complex_stft.real
        imag = complex_stft.imag
        
        return real, imag


class SpatialAudioFeatureExtractor(nn.Module):
    def __init__(self, sr=24000, n_fft=512, hop_length=128, win_length=512, 
                 n_mels=64, fmin=50, fmax=10500, target_frame=16):
        """
        Extract features for spatial audio processing based on SpatialAST model.
        
        Args:
            sr: Sample rate
            n_fft: FFT size
            hop_length: Hop size
            win_length: Window length
            n_mels: Number of mel filter banks
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            target_frame: Target frame length for time dimension
        """
        super(SpatialAudioFeatureExtractor, self).__init__()
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.target_frame = target_frame
        
        # Create STFT extractor
        self.spectrogram_extractor = STFT(
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window='hann', 
            center=True, 
            pad_mode='reflect', 
            freeze_parameters=True
        )
        
        # Create log-mel extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sr, 
            n_fft=n_fft, 
            n_mels=n_mels, 
            fmin=fmin, 
            fmax=fmax, 
            ref=1.0, 
            amin=1e-10, 
            top_db=None, 
            freeze_parameters=True
        )
        
        # Batch normalization for log-mel features
        self.bn = nn.BatchNorm2d(2, affine=False)
        
        # Optional data augmentation transforms
        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)
        

    def apply_reverb(self, waveforms, reverbs):
        """
        Apply reverberation to waveforms using convolution.
        
        Args:
            waveforms: Audio waveforms, (batch_size, channels, samples)
            reverbs: Reverb impulse responses, (batch_size, channels, samples)
        
        Returns:
            Reverberated waveforms, (batch_size, channels, samples)
        """
        # Apply FFT convolution and trim to original length
        reverberated = torchaudio.functional.fftconvolve(
            waveforms, reverbs, mode='full'
        )[..., :waveforms.shape[-1]]
        
        return reverberated
    
    def extract_features(self, waveforms, reverbs=None, apply_augmentation=False):
        """
        Extract spatial audio features from waveforms.
        
        Args:
            waveforms: Audio waveforms, (batch_size, channels, samples)
            reverbs: Optional reverb impulse responses, (batch_size, channels, samples)
            apply_augmentation: Whether to apply time and frequency masking
        
        Returns:
            Dictionary containing extracted features
        """
        # Apply reverberation if provided
        if reverbs is not None:
            waveforms = self.apply_reverb(waveforms, reverbs)
        
        B, C, T = waveforms.shape
        
        # Flatten batch and channel dimensions for STFT
        waveforms_flat = waveforms.reshape(B * C, T)
        
        # Extract spectrograms
        real, imag = self.spectrogram_extractor(waveforms_flat)
        
        # Compute magnitude spectrograms and reshape
        mag_spec = torch.sqrt(real**2 + imag**2)
        
        # Extract log-mel spectrograms and reshape to (batch, channels, time, freq)
        log_mel = self.logmel_extractor(mag_spec).reshape(B, C, -1, self.n_mels)
        
        # Apply batch normalization
        log_mel = self.bn(log_mel)
        
        # Compute Interaural Phase Difference (IPD) features
        # Assuming binaural format with alternating left-right channels
        if C >= 2:
            IPD = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
            IPD_features = torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1)
            IPD_mel = torch.matmul(IPD_features, self.logmel_extractor.melW.to(waveforms.device))
            
            # Combine log-mel and IPD features
            x = torch.cat([log_mel, IPD_mel.reshape(B, 2, -1, self.n_mels)], dim=1)
        else:
            # For mono audio, just use log-mel features
            x = log_mel
        print("current number of frames", x.shape)        
        # Ensure we have the target number of frames
        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(
                x, 
                (self.target_frame, x.shape[3]), 
                mode="bicubic", 
                align_corners=True
            )
        
        return {
            'real': real,
            'imag': imag,
            'log_mel': log_mel,
            'combined_features': x
        }

    def forward(self, waveforms, reverbs=None, apply_augmentation=False):
        """
        Forward pass for the feature extractor.
        
        Args:
            waveforms: Audio waveforms, (batch_size, channels, samples)
            reverbs: Optional reverb impulse responses, (batch_size, channels, samples)
            apply_augmentation: Whether to apply time and frequency masking
        
        Returns:
            Extracted features ready for the transformer model
        """
        return self.extract_features(waveforms, reverbs, apply_augmentation)['combined_features']
