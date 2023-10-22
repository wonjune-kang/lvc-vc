import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('../')
from utils.perturbations import (
    FrequencyWarp,
    formant_and_pitch_shift,
    peq,
    wav_to_Sound,
)
from utils.utils import (
    get_f0_median_std_representations,
    load_and_resample,
    lowquef_lifter,
)


def create_dataloader(hp, train):
    if train:
        if hp.train.use_wav2vec:
            dataset = VCTK_Wav2Vec(hp=hp, train=train)
        else:
            dataset = VCTK_Spect(hp=hp, train=train)
        return DataLoader(
            dataset=dataset,
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        if hp.train.use_wav2vec:
            dataset = VCTK_Wav2Vec(hp=hp, train=train)
        else:
            dataset = VCTK_Spect(hp=hp, train=train)
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
        self.spect_dir = os.path.join(self.root_dir, self.hp.data.spect_dir)

        if hp.train.use_gmm_emb:
            # Load GMMs fit on each speaker's speaker embeddings.
            self.speaker_embedding_gmms = pickle.load(open(hp.data.speaker_embs_gmm_file, "rb"))
        else:
            # Load each speaker's pre-extracted average speaker embeddings.
            self.avg_speaker_embeddings = pickle.load(open(hp.data.avg_speaker_embs_file, "rb"))

        # Paths to pre-extracted normalized F0 contours.
        self.f0_norm_dir = os.path.join(self.root_dir, hp.data.f0_norm_dir)

        # Load metadata on training or test utterances. Pickle files contain
        # utterance filenames for each speaker.
        if self.train:
            metadata = pickle.load(open(hp.data.seen_speakers_train_utts, "rb"))
        else:
            metadata = pickle.load(open(hp.data.seen_speakers_test_utts, "rb"))

        # ### FOR DEBUGGING: Uncomment to load metadata for only a few speakers
        # #                  for faster testing
        # metadata = {k: metadata[k] for k in list(metadata)[:3]}

        # Load utterance data: speaker IDs, F0 contours, raw audio.
        self.data = self.load_metadata(metadata)

        # Load pre-extracted F0 metadata (median, std dev) for each speaker.
        f0_metadata_pkl = pickle.load(open(self.hp.data.f0_metadata_file, "rb"))
        self.f0_metadata = get_f0_median_std_representations(f0_metadata_pkl)

    def load_metadata(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_utterance_data(self, utt_data):
        raise NotImplementedError()

    def crop_feature(self, feature):
        max_feat_start = feature.shape[1] - self.feat_segment_length - 2
        feat_start = random.randint(0, max_feat_start)
        feat_end = feat_start + self.feat_segment_length
        feature = feature[:, feat_start:feat_end]
        return feature, feat_start, feat_end

    def pad_audio_features(self, audio, f0_norm):
        len_pad_samples = self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)
        audio = np.pad(audio, (0, len_pad_samples), mode='constant')

        len_pad_frames = (len_pad_samples // self.feat_hop_length) + 1

        # Need one-hot vector for F0 features, so make the index correspond
        # to that of unvoiced frames (last index = 256).
        f0_norm = np.pad(f0_norm, ((0, 0), (0, len_pad_frames)), mode='constant')
        f0_norm[-len_pad_frames:, -1] = 1

        return audio, f0_norm, len_pad_samples


class VCTK_Spect(VCTK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Frequency warper for mel spectrograms.
        self.frequency_warper = FrequencyWarp()

        # Flag for whether to use SSC loss.
        self.use_ssc = self.hp.train.use_ssc

        if self.train:
            self.ssc_samples = self.hp.ssc.num_ssc_samples
        else:
            # Take random sample of 1000 utterances for for validation loop only.
            # self.data = random.sample(self.data, 1000)
            self.ssc_samples = 1

        # Segment length for all training samples.
        self.feat_hop_length = self.hp.audio.hop_length
        self.feat_segment_length = self.hp.audio.segment_length // self.feat_hop_length

    def load_metadata(self, metadata):
        dataset = []
        for speaker_id, utterances in tqdm(metadata.items()):
            for relative_path in utterances:
                # Load mel-spectrogram. (80, N).
                spect = np.load(os.path.join(self.spect_dir, f"{relative_path[:-4]}.npy"))

                # Quantized normalized F0 contour. (257, N).
                f0_norm = np.load(os.path.join(self.f0_norm_dir, f"{relative_path[:-4]}.npy"))

                # Raw time domain audio.
                utt_wav_path = os.path.join(self.wav_dir, relative_path)
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

    def __getitem__(self, idx):
        # Get data for baseline utterance.
        utt_data = self.data[idx]
        speaker_id1, audio1, spect1, speaker_feat1, f0_norm1 = self.get_utterance_data(utt_data)

        # Warp low-quefrency liftered spectrogram for training.
        if self.train and self.hp.train.warp_lq:
            lp_warp_ratio = np.random.uniform(0.85, 1.15)
            spect1 = self.frequency_warper.warp_spect_frequency(spect1, lp_warp_ratio)

        audio1 = torch.from_numpy(audio1).unsqueeze(0)
        spect1 = torch.from_numpy(spect1)
        f0_norm1 = torch.from_numpy(f0_norm1)

        audios = [audio1]
        perturbed_audios = []
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

                _, audio_new, spect_new, speaker_feat_new, f0_new = self.get_utterance_data(
                    utt_data_new
                )

                audios.append(audio_new)
                spects.append(spect_new)
                speaker_feats.append(speaker_feat_new)
                f0_norms.append(f0_new)

        return audios, perturbed_audios, spects, speaker_feats, f0_norms

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

        # If training, crop all features to constant length.
        if self.train:
            spect, feat_start, feat_end = self.crop_feature(spect)
            f0_norm = f0_norm[:, feat_start:feat_end]

            audio_len = self.hp.audio.segment_length
            audio_start = feat_start * self.hp.audio.hop_length
            audio = audio[audio_start:audio_start + audio_len]

        return speaker_id, audio, spect, speaker_feat, f0_norm


class VCTK_Wav2Vec(VCTK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We don't need SSC loss when using wav2vec features, so we use
        # self.ssc_samples to pass 1 target utterance during inference
        # (validation) only.
        if self.train:
            self.ssc_samples = 0
        else:
            # Take random sample of 1000 utterances for for validation loop only.
            # self.data = random.sample(self.data, 1000)
            self.ssc_samples = 1

        # Segment length for all training samples.
        self.feat_hop_length = self.hp.audio.wav2vec_hop_length
        self.feat_segment_length = self.hp.audio.segment_length // self.feat_hop_length

    def load_metadata(self, metadata):
        dataset = []
        for speaker_id, utterances in tqdm(metadata.items()):
            for relative_path in utterances:
                # Quantized normalized F0 contour. (257, N).
                f0_norm = np.load(os.path.join(self.f0_norm_dir, f"{relative_path[:-4]}.npy"))

                # Raw time domain audio.
                utt_wav_path = os.path.join(self.wav_dir, relative_path)
                audio = load_and_resample(utt_wav_path, self.hp.audio.sampling_rate)

                utterance_data = {
                    'speaker_id': speaker_id,
                    'relative_path': relative_path,
                    'f0_norm': f0_norm,
                    'audio': audio
                }
                dataset.append(utterance_data)

        return dataset

    def __getitem__(self, idx):
        # Get data for baseline utterance.
        utt_data = self.data[idx]
        speaker_id1, audio1, perturbed_audio1, speaker_feat1, f0_norm1 = self.get_utterance_data(
            utt_data
        )

        audio1 = torch.from_numpy(audio1).unsqueeze(0)
        perturbed_audio1 = torch.from_numpy(perturbed_audio1)
        f0_norm1 = torch.from_numpy(f0_norm1)

        audios = [audio1]
        perturbed_audios = [perturbed_audio1]
        spects = []
        speaker_feats = [speaker_feat1]
        f0_norms = [f0_norm1]

        if not self.train:
            # Choose an utterance from another speaker and get features.
            idx_new = random.randint(0, len(self.data)-1)
            utt_data_new = self.data[idx_new]
            while utt_data_new['speaker_id'] == speaker_id1:
                idx_new = random.randint(0, len(self.data)-1)
                utt_data_new = self.data[idx_new]

            _, audio_new, perturbed_audio_new, speaker_feat_new, f0_new = self.get_utterance_data(
                utt_data_new
            )

            audios.append(audio_new)
            perturbed_audios.append(perturbed_audio_new)
            speaker_feats.append(speaker_feat_new)
            f0_norms.append(f0_new)

        return audios, perturbed_audios, spects, speaker_feats, f0_norms

    def get_utterance_data(self, utt_data):
        speaker_id = utt_data['speaker_id']
        audio = utt_data['audio']
        f0_norm = utt_data['f0_norm']

        # Perturb audio for wav2vec 2.0 features.
        perturbed_audio = self.perturb_audio(audio).astype(np.float32)

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
            audio, f0_norm, len_pad_samples = self.pad_audio_features(audio, f0_norm)
            perturbed_audio = np.pad(perturbed_audio, (0, len_pad_samples), mode='constant')

        # If training, crop all features to constant length.
        if self.train:
            f0_norm, feat_start, feat_end = self.crop_feature(f0_norm)

            audio_start = feat_start * self.feat_hop_length
            audio = audio[audio_start:audio_start + self.hp.audio.segment_length]
            perturbed_audio = perturbed_audio[
                audio_start:audio_start + self.hp.audio.segment_length
            ]

        return speaker_id, audio, perturbed_audio, speaker_feat, f0_norm

    def perturb_audio(self, wav):
        # Random frequency shaping via parametric equalizer (peq).
        wav = torch.from_numpy(wav)
        wav = peq(wav, self.hp.audio.sampling_rate).numpy()

        # Formant and pitch shifting.
        sound = wav_to_Sound(wav, sampling_frequency=self.hp.audio.sampling_rate)
        sound = formant_and_pitch_shift(sound)
        perturbed_wav = sound.values[0]

        return perturbed_wav
