import os
import pickle
import argparse
import numpy as np
from scipy.io import wavfile
import torch
from omegaconf import OmegaConf

from transformers import Wav2Vec2ForPreTraining
from model.ecapa_tdnn import ECAPA_TDNN
from model.generator import Generator as LVC_VC
from utils.utils import *
from utils.perturbations import *


def load_ecapa_tdnn(checkpoint_path, device):
    ecapa_tdnn = ECAPA_TDNN(C=1024).eval().to(device)
    ecapa_checkpoint = torch.load(checkpoint_path, map_location=device)

    new_state_dict = {}
    for k, v in ecapa_checkpoint.items():
        if 'speaker_encoder' in k:
            key = k.replace('speaker_encoder.', '')
            new_state_dict[key] = ecapa_checkpoint[k]

    ecapa_tdnn.load_state_dict(new_state_dict)
    return ecapa_tdnn

def load_wav2vec2(device):
    wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval().to(device)
    return wav2vec2

def load_lvc_vc(checkpoint_path, hp, device):
    model = LVC_VC(hp).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval(inference=True)
    
    return model


class LVC_VC_Inference():
    def __init__(
            self,
            hp,
            lvc_vc_chkpt,
            speaker_encoder_chkpt,
            seen_speaker_emb_gmms_pkl,
            seen_speaker_f0_metadata_pkl,
            device
        ):

        # Model hyperparameters.
        self.hp = hp

        # Device.
        self.device = device

        # Define VC model components.
        self.lvc_vc = load_lvc_vc(lvc_vc_chkpt, self.hp, self.device)
        self.speaker_encoder = load_ecapa_tdnn(speaker_encoder_chkpt, self.device)
        self.wav2vec2 = load_wav2vec2(self.device)

        # Define hyperparameters.
        self.fs = self.hp.audio.sampling_rate
        self.fmin = self.hp.audio.mel_fmin
        self.fmax = self.hp.audio.mel_fmax
        self.n_mels = self.hp.audio.n_mel_channels
        self.nfft = self.hp.audio.filter_length
        self.win_length = self.hp.audio.win_length
        self.hop_length = self.hp.audio.hop_length

        # Speaker embeddings and F0 metadata for seen speakers.
        self.seen_speaker_emb_gmms = pickle.load(open(seen_speaker_emb_gmms_pkl, 'rb'))
        self.seen_speaker_f0_metadata = pickle.load(open(seen_speaker_f0_metadata_pkl, 'rb'))

    def perturb_audio(self, wav):
        # Random frequency shaping via parametric equalizer (peq).
        wav = torch.from_numpy(wav)
        wav = peq(wav, self.hp.audio.sampling_rate).numpy()

        # Formant and pitch shifting.
        sound = wav_to_Sound(wav, sampling_frequency=self.hp.audio.sampling_rate)
        sound = formant_and_pitch_shift(sound)
        perturbed_wav = sound.values[0]
        
        return perturbed_wav.astype(np.float32)

    def extract_features(self, source_audio, target_audio, source_seen, target_seen, source_id, target_id):
        source_audio_tensor = torch.from_numpy(source_audio).unsqueeze(0).to(self.device)
        target_audio_tensor = torch.from_numpy(target_audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract wav2vec 2.0 features.
            wav2vec2_outputs = self.wav2vec2(source_audio_tensor, output_hidden_states=True)
            wav2vec2_feat = wav2vec2_outputs.hidden_states[12] # (1, N, 1024)
            wav2vec2_feat = wav2vec2_feat.permute((0,2,1)) # (1, 1024, N)
            wav2vec2_feat = wav2vec2_feat.detach().squeeze().cpu().numpy() # (1024, N)

        # Extract source utterance's normalized F0 contour.
        if source_seen:
            src_f0_median = self.seen_speaker_f0_metadata[source_id]['median']
            src_f0_std = self.seen_speaker_f0_metadata[source_id]['std']
        else:
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
            int(16000*0.025), # frame length of wav2vec 2.0
            int(16000*0.02), # hop length of wav2vec 2.0
        )

        # Transpose to make (257, N) and crop at end to match wav2vec features.
        f0_norm = f0_norm.T[:,:wav2vec2_feat.shape[1]].astype(np.float32)

        # Extract target speaker's speaker embedding.
        if target_seen:
            target_emb = self.seen_speaker_emb_gmms[target_id].means_.astype(np.float32)[0,:]
        else:
            target_emb = self.speaker_encoder(target_audio_tensor, aug=False).detach().squeeze().cpu().numpy()

        # Extract target speaker's quantized median F0.
        if target_seen:
            target_f0_median = self.seen_speaker_f0_metadata[target_id]['median']
        else:
            target_f0_median, _ = extract_f0_median_std(
                target_audio,
                self.fs,
                self.win_length,
                self.hop_length
            )
        target_f0_median = quantize_f0_median(target_f0_median).astype(np.float32)

        # Store all features in dictionary.
        vc_features = {
            'wav2vec2_feat': wav2vec2_feat,
            'source_f0_norm': f0_norm,
            'target_emb': target_emb,
            'target_f0_median': target_f0_median
        }

        return vc_features
    
    def run_inference(self, source_audio, target_audio, source_seen, target_seen, source_id, target_id):
        # Extract all features needed for conversion.
        vc_features = self.extract_features(
            source_audio,
            target_audio,
            source_seen,
            target_seen,
            source_id,
            target_id
        )

        source_wav2vec2_feat = torch.from_numpy(vc_features['wav2vec2_feat']).unsqueeze(0)
        source_f0_norm = torch.from_numpy(vc_features['source_f0_norm']).unsqueeze(0)
        target_emb = torch.from_numpy(vc_features['target_emb']).unsqueeze(0)
        target_f0_median = torch.from_numpy(vc_features['target_f0_median']).unsqueeze(0)

        # Concatenate features to feed into model.
        noise = torch.randn(1, self.hp.gen.noise_dim, source_wav2vec2_feat.size(2)).to(self.device)
        content_feature = torch.cat((source_wav2vec2_feat, source_f0_norm), dim=1).to(self.device)
        speaker_feature = torch.cat((target_emb, target_f0_median), dim=1).to(self.device)
        
        # Perform conversion and rescale power to match source.
        vc_audio = self.lvc_vc(content_feature, noise, speaker_feature).detach().squeeze().cpu().numpy()
        vc_audio = rescale_power(source_audio, vc_audio)

        return vc_audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config/config.yaml",
                        help="yaml file for configuration")
    parser.add_argument('-p', '--lvc_vc_weights', type=str, default="weights/lvc_vc_wav2vec_ecapa.pt",
                        help="path to LVC-VC model weights")
    parser.add_argument('-e', '--se_weights', type=str, default="weights/ecapa_tdnn_pretrained.pt",
                        help="path to speaker encoder model weights")
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help="index of home GPU device")
    parser.add_argument('-s', '--source_file', type=str, required=True,
                        help="source utterance file")
    parser.add_argument('-t', '--target_file', type=str, required=True,
                        help="target utterance file")
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help="output file name")
    args = parser.parse_args()

    # Select device for running models.
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Set up LVC-VC inferencer.
    hp = OmegaConf.load(args.config)
    lvc_vc_inferencer = LVC_VC_Inference(
        hp=hp,
        lvc_vc_chkpt=args.lvc_vc_weights,
        speaker_encoder_chkpt=args.se_weights,
        seen_speaker_emb_gmms_pkl='/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/metadata/ecapa_tdnn_emb_gmms_all.pkl',
        seen_speaker_f0_metadata_pkl='/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/metadata_orig/speaker_f0_metadata.pkl',
        device=device
    )

    # Load source and target audio for conversion.
    source_audio = load_and_resample(args.source_file, hp.audio.sampling_rate)
    target_audio = load_and_resample(args.target_file, hp.audio.sampling_rate)

    # Run voice conversion and write file.
    # By setting source_seen=False and target_seen=False, we are running
    # inference as if both source and target speakers were unseen (zero-shot).
    vc_audio = lvc_vc_inferencer.run_inference(
        source_audio=source_audio,
        target_audio=target_audio,
        source_seen=False,
        target_seen=False,
        source_id=None,
        target_id=None
    )
    wavfile.write(args.output_file, hp.audio.sampling_rate, vc_audio)
    
    print(f"Converted audio written to {args.output_file}.")