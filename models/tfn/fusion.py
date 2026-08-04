import einops
import torch
import torch.nn as nn


class ClassEmbeddingModel(nn.Module):
    def __init__(self, num_categories: int, embed_dim: int, device, freeze_zero_embed=True):
        super().__init__()
        self.device = device
        self.default_idx = num_categories
        self.embed_layer = nn.Embedding(num_categories + 1, embed_dim)

        with torch.no_grad():
            nn.init.normal_(self.embed_layer.weight[:-1], mean=0.0, std=0.02)
            nn.init.zeros_(self.embed_layer.weight[-1])

        if freeze_zero_embed:
            self.embed_layer.weight.data[-1].requires_grad = False

    def forward(self, relation_dict: dict) -> torch.Tensor:
        feats = relation_dict['feats']
        batch_size, num_objects, num_frames, _ = feats.shape

        cids = torch.tensor([
            int(cid_dict['cid'][0]) if 'cid' in cid_dict else self.default_idx
            for cid_dict in relation_dict['cids']
        ], device=self.device)

        object_embeds = self.embed_layer(cids)
        class_embeds = einops.repeat(
            object_embeds,
            "obj feat -> bs obj frame feat",
            bs=batch_size,
            frame=num_frames,
        )
        exist_mask = (feats != 0).any(dim=-1, keepdim=True).to(class_embeds.dtype).to(self.device)
        class_embeds = class_embeds * exist_mask
        return einops.rearrange(class_embeds, "1 obj frame feat -> obj frame feat")


class FusionModel(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        del num_heads
        self.text_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.LayerNorm(dim),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, object_feature, class_embeds):
        projected_class = self.text_proj(class_embeds)
        fused_queries = self.fusion_mlp(torch.cat([object_feature, projected_class], dim=-1))
        return object_feature + self.gate * fused_queries
