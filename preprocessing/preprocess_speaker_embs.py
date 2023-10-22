import os
import pickle
import sys

import numpy as np
import torch
import torchaudio
from sklearn import mixture
from tqdm import tqdm

sys.path.append("../")
from model.ecapa_tdnn import ECAPA_TDNN
from model.ResNetSE34L import MainModel as ResNetModel
from utils.utils import *


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


# Set GPU index.
gpu_idx = 0
device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

# Choose model type. 0 for Fast ResNet-34, 1 for ECAPA-TDNN.
model_type = 1

# Load ResNet speaker encoder model.
if model_type == 0:
    checkpoint_path = "../weights/resnet34sel_pretrained.pt"
    speaker_encoder = load_resnet_encoder(checkpoint_path, device)
    print("Loaded Fast ResNet-34.\n")

# Load ECAPA-TDNN.
elif model_type == 1:
    checkpoint_path = "../weights/ecapa_tdnn_pretrained.pt"
    speaker_encoder = load_ecapa_tdnn(checkpoint_path, device)
    print("Loaded ECAPA-TDNN.\n")

else:
    raise Exception("Invalid model type.")

# For each speaker in VCTK dataset, fit Gaussian to the speaker's utterances
data_dir = "/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/wav16"
speakers = os.listdir(data_dir)

# Extract speaker embeddings.
avg_speaker_embs = {}
resnet_emb_gmms = {}
for speaker_id in tqdm(speakers):
    speaker_utt_path = os.path.join(data_dir, speaker_id)
    relative_utt_paths = os.listdir(speaker_utt_path)

    utt_embeddings = []
    for utt in relative_utt_paths:
        utt_path = os.path.join(speaker_utt_path, utt)
        utt_audio, sr = torchaudio.load(utt_path)
        utt_audio = utt_audio.to(device)

        if model_type == 0:
            utt_emb = speaker_encoder(utt_audio).detach().cpu().squeeze().numpy()
        elif model_type == 1:
            utt_emb = speaker_encoder(utt_audio, aug=False).detach().cpu().squeeze().numpy()
        else:
            raise Exception("Invalid model type.")

        utt_embeddings.append(utt_emb)

    utt_embeddings = np.stack(utt_embeddings)

    gmm_dvector = mixture.GaussianMixture(n_components=1, covariance_type="diag")
    gmm_dvector.fit(utt_embeddings)

    # Dictionary mapping from speaker IDs to (fixed) average speaker embeddings.
    avg_speaker_embs[speaker_id] = np.mean(utt_embeddings, axis=0)

    # Dictionary mapping from speaker IDs to 1 component GMMs fit on the
    # speaker embeddings.
    resnet_emb_gmms[speaker_id] = gmm_dvector

# Save pickle files.
avg_emb_pickle_file = "/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/metadata/ecapa_tdnn_avg_embs_all.pkl"
with open(avg_emb_pickle_file, 'wb') as f:
    pickle.dump(avg_speaker_embs, f)
    print(f"Average speaker embedding saved to {avg_emb_pickle_file}.")

emb_gmm_pickle_file = "/u/wjkang/data/VCTK-Corpus/VCTK-Corpus/metadata/ecapa_tdnn_emb_gmms_all.pkl"
with open(emb_gmm_pickle_file, 'wb') as f:
    pickle.dump(resnet_emb_gmms, f)
    print(f"Speaker embedding Gaussians saved to {emb_gmm_pickle_file}.")
