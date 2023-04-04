import os
import argparse
import pickle
import numpy as np
from scipy.io import wavfile
import torch
from omegaconf import OmegaConf

from utils.stft import TacotronSTFT
from model.ResNetSE34L import MainModel as ResNetModel
from model.generator import Generator as LVC_VC
from model.f0_predictor import F0PredictorNet
from utils.utils import *
from inference import LVC_VC_Inference


def load_f0_predictor(checkpoint_path, device):
    model = F0PredictorNet(embedding_dim=512).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    return model


class LVC_VC_Anonymize(LVC_VC_Inference):
    def __init__(self,
        f0_pred_chkpt,
        speaker_emb_gmm_file,

        **kwargs
    ):
        super().__init__(**kwargs)

        # GMM fit on large dataset to sample speaker embeddings from.
        self.speaker_emb_gmm = pickle.load(open(speaker_emb_gmm_file, 'rb'))

        # Network to predict approx. F0 from speaker embedding.
        self.f0_predictor = load_f0_predictor(f0_pred_chkpt, self.device)

    def sample_embeddings(self, num_candidates=100):
        # Sample a target speaker embedding from the GMM.
        target_embs, _ = self.speaker_emb_gmm.sample(num_candidates)
        target_embs = target_embs.astype(np.float32)
        return target_embs

    def extract_features_anon(self, source_audio, target_emb):
        # Extract source utterance's mel spectrogram.
        source_spect = extract_mel_spectrogram(source_audio, self.stft)

        # Low-quefrency liftering.
        lowquef_liftered = lowquef_lifter(source_spect)

        # Extract source utterance's normalized F0 contour.
        src_f0_median, src_f0_std = extract_f0_median_std(
            source_audio,
            self.fs,
            self.win_length,
            self.hop_length
        )
        f0_norm = get_f0_norm(
            source_audio,
            src_f0_median,
            src_f0_std,
            self.fs,
            self.win_length,
            self.hop_length
        )

        # Transpose to make (257, N) and crop last sample at end to match spectrogram.
        f0_norm = f0_norm.T[:,:source_spect.shape[1]].astype(np.float32)

        # Compute the approximate F0 median value (in Hz) corresponding to the
        # sampled embedding and convert to log.
        tgt_f0_hz = self.f0_predictor(target_emb.to(self.device)).detach().cpu().numpy().item()
        tgt_f0_hz = unnormalize_f0(tgt_f0_hz, librosa.note_to_hz('C2'), librosa.note_to_hz('C5'))
        target_f0_median = quantize_f0_median(np.log(tgt_f0_hz)).astype(np.float32)

        # Store all features in dictionary.
        vc_features = {
            'source_lq_spect': lowquef_liftered,
            'source_f0_norm': f0_norm,
            'target_emb': target_emb,
            'target_f0_median': target_f0_median
        }

        return vc_features


    def perform_anonymization(self, source_audio, target_emb):
        # Extract all features needed for conversion.
        vc_features = self.extract_features_anon(source_audio, target_emb)

        source_lq_spect = torch.from_numpy(vc_features['source_lq_spect']).unsqueeze(0)
        source_f0_norm = torch.from_numpy(vc_features['source_f0_norm']).unsqueeze(0)
        target_emb = vc_features['target_emb']
        target_f0_median = torch.from_numpy(vc_features['target_f0_median']).unsqueeze(0)

        # Concatenate features to feed into model.
        noise = torch.randn(1, self.hp.gen.noise_dim, source_lq_spect.size(2)).to(self.device)
        content_feature = torch.cat((source_lq_spect, source_f0_norm), dim=1).to(self.device)
        speaker_feature = torch.cat((target_emb, target_f0_median), dim=1).to(self.device)
        
        # Perform conversion and rescale power to match source.
        vc_audio = self.lvc_vc(content_feature, noise, speaker_feature).detach().squeeze().cpu().numpy()
        vc_audio = rescale_power(source_audio, vc_audio)

        return vc_audio