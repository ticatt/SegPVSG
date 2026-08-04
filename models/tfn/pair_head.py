import einops
import torch
import torch.nn as nn


class ObjectEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        hidden_dim=512,
        output_dim=256,
        num_heads=8,
        num_layers=2,
    ):
        super().__init__()
        del output_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class PairProposalNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.pair_ffn = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoded_subjects, encoded_objects):
        sub_tokens = encoded_subjects.max(dim=1).values
        obj_tokens = encoded_objects.max(dim=1).values
        num_objects = obj_tokens.size(0)

        sub_pair = einops.repeat(sub_tokens, "sub feat -> sub obj feat", obj=num_objects)
        obj_pair = einops.repeat(obj_tokens, "obj feat -> sub obj feat", sub=num_objects)
        pair_feats = torch.cat([sub_pair, obj_pair], dim=-1)
        pair_matrix = einops.rearrange(self.pair_ffn(pair_feats), "sub obj 1 -> sub obj")
        pair_matrix.fill_diagonal_(0)
        return pair_matrix
