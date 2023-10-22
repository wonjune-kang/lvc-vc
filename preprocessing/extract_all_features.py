import os
import pickle
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append("../")
from utils.stft import TacotronSTFT
from utils.utils import extract_mel_spectrogram, get_f0_norm, load_and_resample


def get_speaker_utterance_paths(speaker_base_path):
    all_utterance_paths = []

    utterance_files = sorted(os.listdir(speaker_base_path))
    for utt_file in utterance_files:
        utt_path = os.path.join(speaker_base_path, utt_file)
        all_utterance_paths.append(utt_path)

    return all_utterance_paths


config = '../config/config_wav2vec_ecapa_c32.yaml'  # CHANGE AS NEEDED
hp = OmegaConf.load(config)

# Define hyperparameters.
fs = hp.audio.sampling_rate
fmin = hp.audio.mel_fmin
fmax = hp.audio.mel_fmax
n_mels = hp.audio.n_mel_channels
nfft = hp.audio.filter_length

# Wav2vec 2.0 features are extracted with 25 ms window length and 20 ms hop.
wav2vec_window_len = int(16000 * 0.025)
wav2vec_hop_len = int(16000 * 0.02)

# Use spectrogram configurations.
spect_window_len = hp.audio.win_length
spect_hop_len = hp.audio.hop_length

# If using wav2vec features, we don't need to extract mel spectrograms.
extract_mel_flag = True

# Create object for computing STFT and mel spectrograms.
stft = TacotronSTFT(
    nfft,
    spect_hop_len,
    spect_window_len,
    n_mels,
    fs,
    fmin,
    fmax,
    center=False,
    device='cpu'
)

# Read all speakers' median and std F0 statistics.
# CHANGE METADATA FILE AS NEEDED.
f0_metadata_file = '/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/metadata/speaker_f0_metadata.pkl'
with open(f0_metadata_file, 'rb') as f:
    f0_median_std_info = pickle.load(f)

# Specify directories to save extracted files.
# MAKE SURE TO CHANGE THESE AS NEEDED!
wav_dir = os.path.join('..', hp.data.root_dir, hp.data.wav_dir)
spect_dir = os.path.join('..', hp.data.root_dir, 'spect')
spect_f0_norm_dir = os.path.join('..', hp.data.root_dir, 'f0_norm')
wav2vec_f0_norm_dir = os.path.join('..', hp.data.root_dir, 'f0_norm_wav2vec')

# Process all utterances from all speakers.
speaker_ids = sorted(os.listdir(wav_dir))
for speaker_id in speaker_ids:
    print(f"Processing {speaker_id}...")

    # Directory for speaker's WAV files.
    speaker_wav_directory = os.path.join(wav_dir, speaker_id)

    # Create directories for storing utterance-wise speaker feature data.
    spect_save_dir = os.path.join(spect_dir, speaker_id)
    spect_f0_norm_save_dir = os.path.join(spect_f0_norm_dir, speaker_id)
    wav2vec_f0_norm_save_dir = os.path.join(spect_f0_norm_dir, speaker_id)
    os.makedirs(spect_save_dir, exist_ok=True)
    os.makedirs(spect_f0_norm_save_dir, exist_ok=True)
    os.makedirs(wav2vec_f0_norm_save_dir, exist_ok=True)

    utterance_files = get_speaker_utterance_paths(speaker_wav_directory)
    for i, utt_file_path in enumerate(tqdm(utterance_files)):
        save_filename = os.path.split(utt_file_path)[1][:-4] + '.npy'

        # Load time domain signal.
        y = load_and_resample(utt_file_path, fs)

        # Extract mel spectrogram.
        spect = extract_mel_spectrogram(y, stft)  # (80, N)
        np.save(
            os.path.join(spect_save_dir, save_filename),
            spect.astype(np.float32),
            allow_pickle=False
        )

        # Compute and save one-hot normalized log F0.
        # Get F0 median and std dev info for current speaker.
        f0_median = f0_median_std_info[speaker_id]['median']
        f0_std = f0_median_std_info[speaker_id]['std']

        # Extract F0 metadata matching spectrograms.
        # Get one-hot representations for normalized log F0 contour.
        f0_norm = get_f0_norm(y, f0_median, f0_std, fs, spect_window_len, spect_hop_len).T

        # Sometimes, there is 1 more F0 frame extracted because of window
        # processing mismatches, so we crop to match the spectrogram length.
        f0_norm = f0_norm[:, :spect.shape[1]]  # (257, N)

        np.save(
            os.path.join(spect_f0_norm_save_dir, save_filename),
            f0_norm.astype(np.float32),
            allow_pickle=False
        )

        # Extract F0 metadata matching wav2vec features
        # (different window and hop length from spectrograms).
        f0_norm = get_f0_norm(y, f0_median, f0_std, fs, wav2vec_window_len, wav2vec_hop_len).T

        np.save(
            os.path.join(wav2vec_f0_norm_save_dir, save_filename),
            f0_norm.astype(np.float32),
            allow_pickle=False
        )
