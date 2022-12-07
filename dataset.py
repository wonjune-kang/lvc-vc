import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append('../')
from utils.utils import load_and_resample, lowquef_lifter, get_f0_median_std_representations
from utils.perturbations import FrequencyWarp


def create_dataloader(hp, train):
    if train:
        dataset = VCTK(hp, train)
        return DataLoader(
            dataset=dataset,
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        dataset = VCTK(hp, train)
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=False
        )


class VCTK(Dataset):
    def __init__(self, hp, train):
        random.seed(hp.train.seed)

        self.hp = hp
        self.train = train

        self.root_dir = hp.data.root_dir
        self.wav_dir = os.path.join(self.root_dir, hp.data.wav_dir)

        if hp.train.use_gmm_emb:
            # Load GMMs fit on each speaker's speaker embeddings.
            self.speaker_embedding_gmms = pickle.load(open(hp.data.speaker_embs_gmm_file, "rb"))
        else:
            # Load each speaker's pre-extracted average speaker embeddings.
            self.avg_speaker_embeddings = pickle.load(open(hp.data.avg_speaker_embs_file, "rb"))

        # Paths to pre-extracted spectrograms and normalized F0 contours.
        self.spect_dir = os.path.join(self.root_dir, hp.data.spect_dir)
        self.f0_norm_dir = os.path.join(self.root_dir, hp.data.f0_norm_dir)

        # Load metadata on training or test utterances. Pickle files contain
        # utterance filenames for each speaker.
        if self.train:
            metadata = pickle.load(open(hp.data.seen_speakers_train_utts, "rb"))
        else:
            metadata = pickle.load(open(hp.data.seen_speakers_test_utts, "rb"))
        metadata = {k: metadata[k] for k in list(metadata)[:5]} ### FOR DEBUGGING

        # Load utterance data: speaker IDs, mel spectrograms, F0 contours, raw audio.
        self.data = self.load_metadata(metadata)

        # Frequency warper for mel spectrograms.
        self.frequency_warper = FrequencyWarp()

        # Flag for whether to use SSC loss.
        self.use_ssc = self.hp.train.use_ssc

        # Take random sample of 1000 utterances for for validation loop only.
        if self.train:
            self.ssc_samples = self.hp.ssc.num_ssc_samples
        else:
            # self.data = random.sample(self.data, 1000)
            self.ssc_samples = 1

        # Load pre-extracted F0 metadata (median, std dev) for each speaker.
        f0_metadata_pkl = pickle.load(open(self.hp.data.f0_metadata_file, "rb"))
        self.f0_metadata = get_f0_median_std_representations(f0_metadata_pkl)

        # Segment length for all training samples.
        self.feat_segment_length = hp.audio.segment_length // hp.audio.hop_length

    def load_metadata(self, metadata):
        dataset = []
        for speaker_id, utterances in tqdm(metadata.items()):
            for relative_path in utterances:
                # Load mel-spectrogram.
                # NOTE: Features were saved as (N, 80) for AutoVC, but need
                # to be in (80, N) for UnivNet, so transpose when reading.
                spect = np.load(os.path.join(self.spect_dir, relative_path)).T  # (80, N)
                
                # Quantized normalized F0 contour.
                f0_norm = np.load(os.path.join(self.f0_norm_dir, relative_path)).T  # (257, N)
                
                # Raw time domain audio.
                utt_wav_path = os.path.join(self.wav_dir, f"{relative_path[:-4]}.wav")
                audio = load_and_resample(utt_wav_path, self.hp.audio.sampling_rate)

                utterance_data = {
                    'speaker_id': speaker_id,
                    'relative_path': relative_path,
                    'spect': spect,
                    'f0_norm': f0_norm,
                    'audio': audio
                }
                dataset.append(utterance_data)

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data for baseline utterance.
        utt_data = self.data[idx]
        speaker_id1, audio1, spect1, speaker_feat1, f0_norm1 = self.get_utterance_data(utt_data)

        # Warp low-quefrency liftered spectrogram for training.
        if self.train and self.hp.train.warp_lp:
            lp_warp_ratio = np.random.uniform(0.85, 1.15)
            spect1 = self.frequency_warper.warp_spect_frequency(spect1, lp_warp_ratio)

        audio1 = torch.from_numpy(audio1).unsqueeze(0)
        spect1 = torch.from_numpy(spect1)
        # speaker_feat1 = torch.tensor(speaker_feat1)
        f0_norm1 = torch.from_numpy(f0_norm1)

        audios = [audio1]
        spects = [spect1]
        speaker_feats = [speaker_feat1]
        f0_norms = [f0_norm1]

        if self.use_ssc or not self.train:
            for _ in range(self.ssc_samples):
                # Choose an utterance from another speaker and get features.
                idx_new = random.randint(0, len(self.data)-1)
                utt_data_new = self.data[idx_new]
                while utt_data_new['speaker_id'] == speaker_id1:
                    idx_new = random.randint(0, len(self.data)-1)
                    utt_data_new = self.data[idx_new]
                
                _, audio_new, spect_new, speaker_feat_new, f0_new = self.get_utterance_data(utt_data_new)

                audios.append(audio_new)
                spects.append(spect_new)
                speaker_feats.append(speaker_feat_new)
                f0_norms.append(f0_new)

        return audios, spects, speaker_feats, f0_norms

    def get_utterance_data(self, utt_data):
        speaker_id = utt_data['speaker_id']
        audio = utt_data['audio']
        spect = utt_data['spect']
        f0_norm = utt_data['f0_norm']

        # Get speaker embedding.
        if self.hp.train.use_gmm_emb:
            # Either sample from GMM...
            speaker_emb, _ = self.speaker_embedding_gmms[speaker_id].sample(1)
            speaker_emb = torch.from_numpy(speaker_emb.astype(np.float32)).squeeze(0)
        else:
            # ...or use a pre-extracted mean embedding.
            speaker_emb = torch.tensor(self.avg_speaker_embeddings[speaker_id])

        # Combine all speaker features.
        f0_median = torch.from_numpy(self.f0_metadata[speaker_id]['median'])
        speaker_feat = torch.cat((speaker_emb, f0_median), dim=0)

        # Pad audio and corresponding features if utterance is too short.
        if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            audio, spect, f0_norm = self.pad_audio_features(audio, spect, f0_norm)

        # Low-quefrency lifter original spectrogram to remove harmonics.
        spect = lowquef_lifter(spect)

        # audio = torch.from_numpy(audio).unsqueeze(0)
        # feature = torch.from_numpy(feature)
        # f0 = torch.from_numpy(f0)

        # If training, crop all features to constant length.
        if self.train:
            spect, feat_start, feat_end = self.crop_feature(spect)
            f0_norm = f0_norm[:, feat_start:feat_end]

            audio_len = self.hp.audio.segment_length
            audio_start = feat_start * self.hp.audio.hop_length
            audio = audio[audio_start:audio_start + audio_len]

        return speaker_id, audio, spect, speaker_feat, f0_norm

    def crop_feature(self, feature):
        max_feat_start = feature.shape[1] - self.feat_segment_length - 1
        feat_start = random.randint(0, max_feat_start)
        feat_end = feat_start + self.feat_segment_length
        feature = feature[:, feat_start:feat_end]

        return feature, feat_start, feat_end

    def pad_audio_features(self, audio, spect, f0_norm):
        len_pad_samples = self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)
        audio = np.pad(audio, (0, len_pad_samples), mode='constant')

        len_pad_frames = (len_pad_samples // self.hp.audio.hop_length) + 1
        spect = np.pad(spect, ((0,0), (0,len_pad_frames)), mode='constant')
        
        # Need one-hot vector for F0 features, so make the index correspond
        # to that of unvoiced frames (last index = 256).
        f0_norm = np.pad(f0_norm, ((0,0), (0,len_pad_frames)), mode='constant')
        f0_norm[-len_pad_frames:, -1] = 1

        return audio, spect, f0_norm