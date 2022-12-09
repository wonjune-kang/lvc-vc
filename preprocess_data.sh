#!/bin/bash

cd preprocessing

# Resample VCTK to 16 kHz.
python resample_vctk.py

# Extract speaker-wise F0 median and std dev
python extract_f0_metadata.py

# Extract mel spectrograms and normalized F0 contours
python extract_all_features.py

# Split VCTK corpus into 99 seen and 10 unseen speakers.
# Further split train speakers' utterance into train and test.
python split_data.py

# Fit GMMs to each speaker's embeddings and compute average embeddings. 
python preprocess_speaker_embs.py