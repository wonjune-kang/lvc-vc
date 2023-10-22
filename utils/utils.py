import subprocess

import librosa
import numpy as np
import torch
from scipy.fft import dct, idct


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')


def load_and_resample(utterance_path, new_sr):
    y, sr = librosa.load(utterance_path, sr=None)
    if new_sr != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
        y = np.clip(y, -1.0, 32767.0/32768.0)
    return y


def extract_mel_spectrogram(wav, stft):
    wav = torch.from_numpy(wav).unsqueeze(0)
    mel = stft.mel_spectrogram(wav)
    mel = mel.squeeze(0).numpy()
    return mel


def lowquef_lifter(mel, quef_order=20):
    mel_cepstrum = dct(mel, axis=0)
    mel_cepstrum[quef_order:, :] = 0.0
    liftered_mel = idct(mel_cepstrum, axis=0)
    return liftered_mel


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def quantize(value, bins, num_bins):
    one_hot_idx = np.minimum(np.digitize(value, bins)-1, num_bins-1)
    one_hot_representation = one_hot(one_hot_idx, num_bins).astype(np.float32)
    return one_hot_representation


def extract_utterance_log_f0(y, sr, frame_len_samples, hop_len_samples, voiced_prob_cutoff=0.2):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C5'),
        sr=sr,
        frame_length=frame_len_samples,
        hop_length=hop_len_samples,
        n_thresholds=5
    )

    f0[np.where(voiced_probs < voiced_prob_cutoff)] = np.nan
    log_f0 = np.log(f0)

    return log_f0


def quantize_f0_norm(y, f0_median, f0_std, fs, win_length, hop_length):
    utt_log_f0 = extract_utterance_log_f0(y, fs, win_length, hop_length)
    log_f0_norm = ((utt_log_f0 - f0_median) / f0_std) / 4.0

    return log_f0_norm


def get_f0_norm(y, f0_median, f0_std, fs, win_length, hop_length, num_f0_bins=256):
    log_f0_norm = quantize_f0_norm(y, f0_median, f0_std, fs, win_length, hop_length) + 0.5

    bins = np.linspace(0, 1, num_f0_bins+1)
    f0_one_hot_idxs = np.digitize(log_f0_norm, bins) - 1
    f0_one_hot = one_hot(f0_one_hot_idxs, num_f0_bins+1)

    return f0_one_hot


def extract_f0_median_std(wav, fs, win_length, hop_length):
    log_f0_vals = extract_utterance_log_f0(wav, fs, win_length, hop_length)
    log_f0_vals = log_f0_vals[~np.isnan(log_f0_vals)]

    log_f0_median = np.median(log_f0_vals).astype(np.float32)
    log_f0_std = np.std(log_f0_vals).astype(np.float32)

    return log_f0_median, log_f0_std


def quantize_f0_median(
        f0_median,
        log_fmin=np.log(librosa.note_to_hz('C2')),
        log_fmax=np.log(librosa.note_to_hz('C5')),
        num_bins=64
):
    bins = np.linspace(log_fmin, log_fmax, num_bins+1)
    median_one_hot = quantize(f0_median, bins, num_bins)
    return median_one_hot


def get_f0_median_std_representations(
    f0_median_std_metadata,
    log_fmin=np.log(librosa.note_to_hz('C2')),
    log_fmax=np.log(librosa.note_to_hz('C5')),
    std_min=0.05,
    std_max=0.35,
    num_bins=64
):

    median_bins = np.linspace(log_fmin, log_fmax, num_bins+1)
    std_bins = np.linspace(std_min, std_max, num_bins+1)

    f0_median_std_quantized = {}
    for speaker_id, f0_metadata in f0_median_std_metadata.items():
        f0_median = f0_metadata['median']
        f0_std = f0_metadata['std']

        median_one_hot = quantize(f0_median, median_bins, num_bins)
        std_one_hot = quantize(f0_std, std_bins, num_bins)

        f0_median_std_quantized[speaker_id] = {
            'median': median_one_hot,
            'std': std_one_hot
        }

    return f0_median_std_quantized


def unnormalize_f0(norm_values, min, max):
    return norm_values * (max - min) + min


def rescale_power(source_wav, vc_wav):
    # Rescale the output.
    source_pow = np.sum(np.power(source_wav, 2.))
    vc_pow = np.sum(np.power(vc_wav, 2.))
    scale = np.sqrt(source_pow / (vc_pow + 1e-6))

    vc_wav *= scale

    return vc_wav
