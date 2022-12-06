import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerSimilarityCriterion(nn.Module):
    def __init__(self, speaker_encoder, device):
        super(SpeakerSimilarityCriterion, self).__init__()
        
        self.speaker_encoder = speaker_encoder
        self.device = device

    def forward(self, vc_audios, src_audios, tgt_speaker_emb):
        vc_speaker_embs = []
        src_speaker_embs = []
        for vc_audio, src_audio in zip(vc_audios, src_audios):
            vc_speaker_emb = self.speaker_encoder(vc_audio)
            src_speaker_emb = self.speaker_encoder(src_audio)
            vc_speaker_embs.append(vc_speaker_emb)
            src_speaker_embs.append(src_speaker_emb)
        
        vc_speaker_embs = torch.stack(vc_speaker_embs).transpose(0, 1) # (B, 8, 512)
        src_speaker_embs = torch.stack(src_speaker_embs).transpose(0, 1) # (B, 8, 512)
        tgt_speaker_embs_exp = tgt_speaker_emb.unsqueeze(1).repeat((1, len(vc_audios), 1))

        pos_loss = 0.0
        neg_loss = 0.0
        pos_labels = torch.ones(vc_speaker_embs.shape[1]).to(self.device)
        neg_labels = -torch.ones(vc_speaker_embs.shape[1]).to(self.device)

        for vc_embs_batch, src_embs_batch, tgt_emb_exp_batch in zip(vc_speaker_embs, src_speaker_embs, tgt_speaker_embs_exp):
            pos_loss += F.cosine_embedding_loss(vc_embs_batch, tgt_emb_exp_batch, pos_labels)
            neg_loss += F.cosine_embedding_loss(vc_embs_batch, src_embs_batch, neg_labels)
        
        pos_loss /= vc_speaker_embs.shape[0]
        neg_loss /= vc_speaker_embs.shape[0]

        return pos_loss, neg_loss