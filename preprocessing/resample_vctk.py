import os
import sys

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append("../")
from utils.utils import load_and_resample

fs = 16000
wav_dir = "../data/VCTK-Corpus/VCTK-Corpus/wav48"
resample_dir = "../data/VCTK-Corpus/VCTK-Corpus/wav16"

os.makedirs(resample_dir, exist_ok=True)

# Process all utterances from all speakers.
speaker_ids = sorted(os.listdir(wav_dir))
for speaker_id in speaker_ids:
    print(f"Processing {speaker_id}...")

    speaker_wav_dir = os.path.join(wav_dir, speaker_id)
    speaker_resample_dir = os.path.join(resample_dir, speaker_id)
    os.makedirs(speaker_resample_dir, exist_ok=True)

    utterance_files = sorted(os.listdir(speaker_wav_dir))
    for i, utt_file in enumerate(tqdm(utterance_files)):
        speaker_utt_path = os.path.join(speaker_wav_dir, utt_file)

        # Load and resample time domain signal.
        resampled_float = load_and_resample(speaker_utt_path, fs)
        resampled_int = (resampled_float*32768.).astype('int16')

        utt_save_path = os.path.join(speaker_resample_dir, utt_file)
        wavfile.write(utt_save_path, fs, resampled_int)