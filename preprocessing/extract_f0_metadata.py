import os
import pickle
import random
import sys

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append("../")
from utils.utils import extract_utterance_log_f0, load_and_resample

random.seed(0)


def get_speaker_utterance_paths(speaker_base_path):
    all_utterance_paths = []

    utterance_files = sorted(os.listdir(speaker_base_path))
    for utt_file in utterance_files:
        utt_path = os.path.join(speaker_base_path, utt_file)
        all_utterance_paths.append(utt_path)

    return all_utterance_paths


def find_speaker_f0_median_std(speaker_utt_path, fs, window, hop, voiced_prob_cutoff):
    frame_len_samples = int(fs * window)
    hop_len_samples = int(fs * hop)

    utterance_files = get_speaker_utterance_paths(speaker_utt_path)

    k = min(50, len(utterance_files))
    utterance_files = random.sample(utterance_files, k)

    all_f0_vals = []
    for utt_file in tqdm(utterance_files):
        try:
            y = load_and_resample(utt_file, fs)
            f0_vals = extract_utterance_log_f0(
                y, fs, frame_len_samples, hop_len_samples, voiced_prob_cutoff
            )
            f0_vals = f0_vals[~np.isnan(f0_vals)]
            all_f0_vals.extend(f0_vals.tolist())
        except:
            pass

    all_f0_vals = np.array(all_f0_vals)
    f0_median = np.median(all_f0_vals).astype(np.float32)
    f0_std = np.std(all_f0_vals).astype(np.float32)

    return f0_median, f0_std


def process_all_speaker_f0(speaker_directory, fs, window, hop, voiced_prob_cutoff=0.2):
    all_speakers = sorted(os.listdir(speaker_directory))
    all_speaker_f0_info = {}

    for speaker_id in all_speakers:
        print(f"Processing speaker {speaker_id}...")
        speaker_utt_path = os.path.join(speaker_directory, speaker_id)
        f0_median, f0_std = find_speaker_f0_median_std(
            speaker_utt_path, fs, window, hop, voiced_prob_cutoff
        )

        current_speaker_f0_info = {
            'median': f0_median,
            'std': f0_std
        }
        all_speaker_f0_info[speaker_id] = current_speaker_f0_info

    return all_speaker_f0_info


# Specify config file.
config = '../config/config_wav2vec_ecapa_c32.yaml'
hp = OmegaConf.load(config)

# Directories for reading audio data and saving extracted F0 information.
data_dir = os.path.join('..', hp.data.root_dir, hp.data.wav_dir)
save_dir = os.path.join('..', hp.data.root_dir, 'metadata')
os.makedirs(save_dir, exist_ok=True)

fs = hp.audio.sampling_rate
window = hp.audio.win_length / hp.audio.sampling_rate
hop = hp.audio.hop_length / hp.audio.sampling_rate

# Compute median and std dev of log F0 statistics for every speaker.
print("Computing F0 median and std dev information for all speakers...")
speaker_f0_info = process_all_speaker_f0(data_dir, fs, window, hop, voiced_prob_cutoff=0.2)

# Save dictionary mapping from speaker IDs to their F0 metadata (median, std).
speaker_f0_metadata_save_path = os.path.join(save_dir, 'speaker_f0_metadata.pkl')
with open(speaker_f0_metadata_save_path, 'wb') as f:
    pickle.dump(speaker_f0_info, f)

print(f"F0 median and std dev metadata saved to {speaker_f0_metadata_save_path}.")
