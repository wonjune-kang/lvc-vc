import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.io import wavfile

from model.generator import Generator as LVC_VC
from model.ResNetSE34L import MainModel as ResNetModel
from utils.stft import TacotronSTFT
from utils.utils import (
    extract_f0_median_std,
    extract_mel_spectrogram,
    get_f0_norm,
    load_and_resample,
    lowquef_lifter,
    quantize_f0_median,
    rescale_power,
)


def load_resnet_encoder(checkpoint_path, device):
    model = ResNetModel(512).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    new_state_dict = {}
    for k, v in checkpoint.items():
        try:
            new_state_dict[k[6:]] = checkpoint[k]
        except KeyError:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    return model


def load_lvc_vc(checkpoint_path, hp, device):
    model = LVC_VC(hp).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except KeyError:
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
            device
    ):

        # Model hyperparameters.
        self.hp = hp

        # Device.
        self.device = device

        # Define VC model.
        self.lvc_vc = load_lvc_vc(lvc_vc_chkpt, self.hp, self.device)
        self.speaker_encoder = load_resnet_encoder(speaker_encoder_chkpt, self.device)

        # Define hyperparameters.
        self.fs = self.hp.audio.sampling_rate
        self.fmin = self.hp.audio.mel_fmin
        self.fmax = self.hp.audio.mel_fmax
        self.n_mels = self.hp.audio.n_mel_channels
        self.nfft = self.hp.audio.filter_length
        self.win_length = self.hp.audio.win_length
        self.hop_length = self.hp.audio.hop_length

        # Create object for computing STFT and mel spectrograms.
        self.stft = TacotronSTFT(
            self.nfft,
            self.hop_length,
            self.win_length,
            self.n_mels,
            self.fs,
            self.fmin,
            self.fmax,
            center=False,
            device='cpu'
        )

    def extract_features(self, source_audio, target_audio):
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
        f0_norm = f0_norm.T[:, :source_spect.shape[1]].astype(np.float32)

        # Extract target speaker's quantized median F0.
        target_f0_median, _ = extract_f0_median_std(
            target_audio,
            self.fs,
            self.win_length,
            self.hop_length
        )
        target_f0_median = quantize_f0_median(target_f0_median).astype(np.float32)

        # Extract target speaker's speaker embedding.
        target_audio = torch.from_numpy(target_audio).unsqueeze(0).to(self.device)
        target_emb = self.speaker_encoder(target_audio).cpu()

        # Store all features in dictionary.
        vc_features = {
            'source_lq_spect': lowquef_liftered,
            'source_f0_norm': f0_norm,
            'target_emb': target_emb,
            'target_f0_median': target_f0_median
        }

        return vc_features

    def run_inference(self, source_audio, target_audio):
        # Extract all features needed for conversion.
        vc_features = self.extract_features(source_audio, target_audio)

        source_lq_spect = torch.from_numpy(vc_features['source_lq_spect']).unsqueeze(0)
        source_f0_norm = torch.from_numpy(vc_features['source_f0_norm']).unsqueeze(0)
        target_emb = vc_features['target_emb']
        target_f0_median = torch.from_numpy(vc_features['target_f0_median']).unsqueeze(0)

        # Concatenate features to feed into model.
        noise = torch.randn(1, self.hp.gen.noise_dim, source_lq_spect.size(2)).to(self.device)
        content_feature = torch.cat((source_lq_spect, source_f0_norm), dim=1).to(self.device)
        speaker_feature = torch.cat((target_emb, target_f0_median), dim=1).to(self.device)

        # Perform conversion and rescale power to match source.
        vc_audio = self.lvc_vc(
            content_feature, noise, speaker_feature
        ).detach().squeeze().cpu().numpy()
        vc_audio = rescale_power(source_audio, vc_audio)

        return vc_audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config/config_spect_c16.yaml",
                        help="yaml file for configuration")
    parser.add_argument('-p', '--lvc_vc_weights', type=str, default="weights/lvc_vc_vctk.pt",
                        help="path to LVC-VC model weights")
    parser.add_argument('-e', '--se_weights', type=str, default="weights/resnet34sel_pretrained.pt",
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
        hp,
        args.lvc_vc_weights,
        args.se_weights,
        device
    )

    # Load source and target audio for conversion.
    source_audio = load_and_resample(args.source_file, hp.audio.sampling_rate)
    target_audio = load_and_resample(args.target_file, hp.audio.sampling_rate)

    # Run voice conversion and write file.
    vc_audio = lvc_vc_inferencer.run_inference(source_audio, target_audio)
    wavfile.write(args.output_file, hp.audio.sampling_rate, vc_audio)

    print(f"Converted audio written to {args.output_file}.")
