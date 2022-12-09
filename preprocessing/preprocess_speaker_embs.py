import os
import pickle
from tqdm import tqdm
import numpy as np
from sklearn import mixture
import torch
import torchaudio

import sys
sys.path.append("../")
from utils.utils import *
from model.ResNetSE34L import MainModel as ResNetModel


def load_resnet_encoder(checkpoint_path, device):
    model = ResNetModel(512).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    new_state_dict = {}
    for k, v in checkpoint.items():
        try:
            new_state_dict[k[6:]] = checkpoint[k]
        except:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    return model


# Set GPU index.
gpu_idx = 1
device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

# Load ResNet speaker encoder model.
resnet_weights_path = "../weights/resnet34sel_pretrained.pt"
resnet_speaker_encoder = load_resnet_encoder(resnet_weights_path, device)

# For each speaker in VCTK dataset, fit Gaussian to the speaker's utterances
data_dir = "../data/VCTK-Corpus/VCTK-Corpus/wav16"
speakers = os.listdir(data_dir)

# We want to fit the GMMs only on the seen speakers' training utterances.
train_utts_file = "../data/VCTK-Corpus/VCTK-Corpus/metadata/seen_speakers_train_utts.pkl"
seen_speaker_train_utts = pickle.load(open(train_utts_file, 'rb'))


# Extract speaker embeddings.
avg_speaker_embs = {}
resnet_emb_gmms = {}
for speaker_id in tqdm(speakers):
    try:
        relative_utt_paths = seen_speaker_train_utts[speaker_id]

        utt_embeddings = []
        for utt in relative_utt_paths:
            utt_path = os.path.join(data_dir, utt)
            utt_audio, sr = torchaudio.load(utt_path)
            utt_audio = utt_audio.to(device)
            utt_emb = resnet_speaker_encoder(utt_audio).detach().cpu().squeeze().numpy()
            utt_embeddings.append(utt_emb)
        
        utt_embeddings = np.stack(utt_embeddings)
        
        gmm_dvector = mixture.GaussianMixture(n_components=1, covariance_type="diag")
        gmm_dvector.fit(utt_embeddings)
        
        # Dictionary mapping from speaker IDs to (fixed) average speaker embeddings.
        avg_speaker_embs[speaker_id] = np.mean(utt_embeddings, axis=0)

        # Dictionary mapping from speaker IDs to 1 component GMMs fit on the
        # speaker embeddings.
        resnet_emb_gmms[speaker_id] = gmm_dvector
    except KeyError:
        pass

# Save pickle files.
with open('../data/VCTK-Corpus/VCTK-Corpus/metadata/avg_speaker_resnet_embs.pkl', 'wb') as f:
    pickle.dump(avg_speaker_embs, f)

with open('../data/VCTK-Corpus/VCTK-Corpus/metadata/resnet_emb_gmms.pkl', 'wb') as f:
    pickle.dump(resnet_emb_gmms, f)