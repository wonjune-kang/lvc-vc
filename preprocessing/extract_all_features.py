import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import OmegaConf

import sys
sys.path.append("../")
from utils.utils import load_and_resample, extract_mel_spectrogram, get_f0_norm
from utils.stft import TacotronSTFT


def get_speaker_utterance_paths(speaker_base_path):
    all_utterance_paths = []

    utterance_files = sorted(os.listdir(speaker_base_path))
    for utt_file in utterance_files:
        utt_path = os.path.join(speaker_base_path, utt_file)
        all_utterance_paths.append(utt_path)
    
    return all_utterance_paths


config = '../config/config_wav2vec_ecapa_c32.yaml'
hp = OmegaConf.load(config)

# Define hyperparameters.
fs = hp.audio.sampling_rate
fmin = hp.audio.mel_fmin
fmax = hp.audio.mel_fmax
n_mels = hp.audio.n_mel_channels
nfft = hp.audio.filter_length

if hp.train.use_wav2vec:
    # Wav2vec 2.0 features are extracted with 25 ms window length and 20 ms hop.
    window_len = int(16000 * 0.025)
    hop_len = int(16000 * 0.02)

    # If using wav2vec features, we don't need to extract mel spectrograms.
    extract_mel_flag = False

else:
    # Use spectrogram configurations.
    window_len = hp.audio.win_length
    hop_len = hp.audio.hop_length

    # If using wav2vec features, we don't need to extract mel spectrograms.
    extract_mel_flag = True

    # Create object for computing STFT and mel spectrograms.
    stft = TacotronSTFT(
        nfft,
        hop_len,
        window_len,
        n_mels,
        fs,
        fmin,
        fmax,
        center=False,
        device='cpu'
    )

# Read all speakers' median and std F0 statistics.
f0_metadata_file = '/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/metadata/speaker_f0_metadata.pkl'
with open(f0_metadata_file, 'rb') as f:
    f0_median_std_info = pickle.load(f)

# For debugging in case of mismatches between spectrogram frame length and
# F0 frame length.
mismatched_utterances = []

wav_dir = os.path.join('..', hp.data.root_dir, hp.data.wav_dir)
spect_dir = os.path.join('..', hp.data.root_dir, 'spect')
f0_norm_dir = os.path.join('..', hp.data.root_dir, hp.data.f0_norm_dir)

# Process all utterances from all speakers.
speaker_ids = sorted(os.listdir(wav_dir))
for speaker_id in speaker_ids:
    print(f"Processing {speaker_id}...")

    # Directory for speaker's WAV files.
    speaker_wav_directory = os.path.join(wav_dir, speaker_id)
    
    # Create directories for storing utterance-wise speaker feature data.
    speaker_spect_save_dir = os.path.join(spect_dir, speaker_id)
    speaker_f0_save_dir = os.path.join(f0_norm_dir, speaker_id)
    os.makedirs(speaker_spect_save_dir, exist_ok=True)
    os.makedirs(speaker_f0_save_dir, exist_ok=True)

    utterance_files = get_speaker_utterance_paths(speaker_wav_directory)
    for i, utt_file_path in enumerate(tqdm(utterance_files)):
        save_filename = os.path.split(utt_file_path)[1][:-4] + '.npy'

        # Load time domain signal.
        y = load_and_resample(utt_file_path, fs)

        # Compute and save mel spectrogram.
        if extract_mel_flag:
            spect = extract_mel_spectrogram(y, stft)  # (80, N)
            np.save(os.path.join(speaker_spect_save_dir, save_filename),
                    spect.astype(np.float32), allow_pickle=False)

        # Save one-hot normalized log F0.
        # Get F0 median and std dev info for current speaker.
        f0_median = f0_median_std_info[speaker_id]['median']
        f0_std = f0_median_std_info[speaker_id]['std']

        # Get one-hot representations for normalized log F0 contour.
        f0_norm = get_f0_norm(y, f0_median, f0_std, fs, window_len, hop_len).T
        
        if f0_norm.shape[1] == spect.shape[1] + 1:
            f0_norm = f0_norm[:, :spect.shape[1]]  # (257, N)
        else:
            mismatched_utterances.append((utt_file_path, spect.shape[0], f0_norm.shape[0]))

        np.save(os.path.join(speaker_f0_save_dir, save_filename),
                f0_norm.astype(np.float32), allow_pickle=False)