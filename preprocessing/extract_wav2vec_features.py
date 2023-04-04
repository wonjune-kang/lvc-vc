import os
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import librosa

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor


def get_speaker_utterance_paths(speaker_base_path):
    all_utterance_paths = []

    utterance_files = sorted(os.listdir(speaker_base_path))
    for utt_file in utterance_files:
        utt_path = os.path.join(speaker_base_path, utt_file)
        all_utterance_paths.append(utt_path)
    
    return all_utterance_paths


# Set up device and configurations.
device = "cuda:0" if torch.cuda.is_available() else "cpu"

config = '../config/config.yaml'
hp = OmegaConf.load(config)

# Define hyperparameters.
sampling_rate = hp.audio.sampling_rate

# Load pretrained XLSR-53 model.
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav2vec_model = wav2vec_model.eval().to(device)

# Process all utterances from all speakers.
wav_dir = os.path.join(hp.data.root_dir, hp.data.wav_dir)
wav2vec_dir = os.path.join(hp.data.root_dir, 'wav2vec_xlsr53')

speaker_ids = sorted(os.listdir(wav_dir))
for speaker_id in speaker_ids:
    print(f"Processing {speaker_id}...")

    # Directory for speaker's WAV files.
    speaker_wav_directory = os.path.join(wav_dir, speaker_id)

    # Create directories for storing utterance-wise speaker feature data.
    speaker_wav2vec_save_dir = os.path.join(wav2vec_dir, speaker_id)
    os.makedirs(speaker_wav2vec_save_dir, exist_ok=True)

    utterance_files = get_speaker_utterance_paths(speaker_wav_directory)
    for i, utt_file_path in enumerate(tqdm(utterance_files)):
        save_filename = os.path.split(utt_file_path)[1][:-4] + '.npy'

        y, sr = librosa.load(utt_file_path, sr=None)
        if sr != sampling_rate:
            raise Exception("Incorrect sampling rate. Must be 16000 kHz.")
        
        features = feature_extractor(
            y,
            return_tensors='pt',
            sampling_rate=sampling_rate
        ).to(device)

        output = wav2vec_model(
            input_values=features['input_values'],
            attention_mask=features['attention_mask'],
            output_hidden_states=True
        )

        # Take output of 12th transformer layer as linguistic features.
        # 0th index is pre-encoder, 1st-24th indices are transformer layers.
        linguistic_feats = output.hidden_states[12].detach().cpu().squeeze().numpy().T # (1024, N)

        # Save linguistic features.
        np.save(os.path.join(speaker_wav2vec_save_dir, save_filename),
                linguistic_feats.astype(np.float32), allow_pickle=False)