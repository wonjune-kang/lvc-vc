import os
import pickle
import random

random.seed(0)


# Create directory to save all metadata to.
metadata_dir = "../data/VCTK-Corpus/VCTK-Corpus/metadata"
os.makedirs(metadata_dir, exist_ok=True)

# Get all speaker IDs.
wav_dir = "../data/VCTK-Corpus/VCTK-Corpus/wav48"
speaker_ids = sorted(os.listdir(wav_dir))

# Sample unseen speakers.
num_unseen = 10
unseen_speakers = sorted(random.sample(speaker_ids, num_unseen))
seen_speakers = set(speaker_ids) - set(unseen_speakers)

# Save unseen speakers to pickle file.
with open(os.path.join(metadata_dir, "unseen_speakers.pkl"), "wb") as f:
    pickle.dump(unseen_speakers, f)


# Get relative paths to all seen speakers' utterances.
seenspeakers2utterances = {}
for speaker_id in sorted(seen_speakers):
    speaker_wav_dir = os.path.join(wav_dir, speaker_id)
    utterances = []
    for wav_file in sorted(os.listdir(speaker_wav_dir)):
        utterances.append(os.path.join(speaker_id, wav_file))

    seenspeakers2utterances[speaker_id] = utterances


# Split seen speakers' utterances into train and test sets.
seenspeakers2train_utts = {}
seenspeakers2test_utts = {}
for speaker_id, utterances in seenspeakers2utterances.items():
    split_idx = int(0.1*len(utterances))
    random.shuffle(utterances)

    train_utts = utterances[:-split_idx]
    test_utts = utterances[-split_idx:]

    seenspeakers2train_utts[speaker_id] = train_utts
    seenspeakers2test_utts[speaker_id] = test_utts


# Save seen speakers' train and test splits to pickle files.
with open(os.path.join(metadata_dir, "seen_speakers_train_utts.pkl"), "wb") as f:
    pickle.dump(seenspeakers2train_utts, f)

with open(os.path.join(metadata_dir, "seen_speakers_test_utts.pkl"), "wb") as f:
    pickle.dump(seenspeakers2test_utts, f)
