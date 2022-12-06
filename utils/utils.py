import random
import subprocess
import numpy as np
from scipy.io.wavfile import read
from scipy.fft import dct, idct
import librosa


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def load_and_resample(utterance_path, new_sr):
    y, sr = librosa.load(utterance_path, sr=None)
    if new_sr != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
        y = np.clip(y, -1.0, 32767.0/32768.0)
    return y

def lowquef_lifter(mel, quef_order=20):
    mel_cepstrum = dct(mel, axis=0)
    mel_cepstrum[quef_order:,:] = 0.0
    liftered_mel = idct(mel_cepstrum, axis=0)
    return liftered_mel

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def quantize(value, bins, num_bins):
    one_hot_idx = np.minimum(np.digitize(value, bins)-1, num_bins-1)
    one_hot_representation = one_hot(one_hot_idx, num_bins).astype(np.float32)
    return one_hot_representation

def extract_utterance_f0(y, sr, frame_len_samples, hop_len_samples, voiced_prob_cutoff=0.2):
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

def quantize_f0(y, hp, f0_mean, f0_std):
    frame_len_samples = int(hp.data.fs * hp.data.window)
    hop_len_samples = int(hp.data.fs * hp.data.hop)
    
    utt_log_f0 = extract_utterance_f0(y, hp.data.fs, frame_len_samples, hop_len_samples)
    log_f0_norm = ((utt_log_f0 - f0_mean) / f0_std) / 4.0
    
    return log_f0_norm

def get_f0_norm(y, hp, f0_mean, f0_std, num_f0_bins=256):
    log_f0_norm = quantize_f0(y, hp, f0_mean, f0_std) + 0.5
    
    bins = np.linspace(0, 1, num_f0_bins+1)
    f0_one_hot_idxs = np.digitize(log_f0_norm, bins) - 1
    f0_one_hot = one_hot(f0_one_hot_idxs, num_f0_bins+1)

    return f0_one_hot

def get_f0_mean_std_representations(f0_mean_std_metadata,
                                    log_fmin=np.log(librosa.note_to_hz('C2')),
                                    log_fmax=np.log(librosa.note_to_hz('C5')),
                                    std_min=0.05, std_max=0.35, num_bins=64):
    
    mean_bins = np.linspace(log_fmin, log_fmax, num_bins+1)
    std_bins = np.linspace(std_min, std_max, num_bins+1)
    
    f0_mean_std_quantized = {}
    for speaker_id, f0_metadata in f0_mean_std_metadata.items():
        f0_mean = f0_metadata['mean']
        f0_std = f0_metadata['std']
        
        mean_one_hot = quantize(f0_mean, mean_bins, num_bins)
        std_one_hot = quantize(f0_std, std_bins, num_bins)

        f0_mean_std_quantized[speaker_id] = {'mean': mean_one_hot,
                                             'std': std_one_hot}
    
    return f0_mean_std_quantized

def get_f0_warp_ratio_representation(linear_warp_ratio,
                                     log_min=-1, # np.log2(0.5)
                                     log_max=1, # np.log2(2.0)
                                     num_bins=256):
    
    bins = np.linspace(log_min, log_max, num_bins+1)
    log_warp_ratio = np.log2(linear_warp_ratio)

    quantized_log_warp_ratio = quantize(log_warp_ratio, bins, num_bins)
    return quantized_log_warp_ratio