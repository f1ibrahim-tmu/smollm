import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
try:
    from VisionMamba.mamba-1p1p1.mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    from mamba_ssm.modules.mamba_simple import Mamba

# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = ((self.img_size[0] - self.patch_size[0]) // stride + 1, (self.img_size[1] - self.patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

# --- Block ---
class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, drop_path=0., mixer_kwargs=None):
        super().__init__()
        if mixer_kwargs is None:
            mixer_kwargs = {}
        self.mixer = mixer_cls(dim, **mixer_kwargs)
        self.norm = norm_cls(dim)
        self.drop_path = nn.Identity()  # DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states: Tensor, residual: Tensor = None):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


# --- Simple Mixer (can be replaced with Mamba or other SSMs) ---
class SimpleMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x, **kwargs):
        return self.mlp(x)

# --- create_block using Mamba as the mixer ---
def create_block(d_model, norm_epsilon=1e-5, drop_path=0., layer_idx=None, mixer_kwargs=None):
    mixer_cls = Mamba
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon)
    return Block(d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path, mixer_kwargs=mixer_kwargs)

# --- VisionMamba (minimal, with Mamba mixer) ---
class VisionMamba(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, depth=4, embed_dim=768, channels=3, drop_path_rate=0.1, d_state=16, **mamba_kwargs):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            create_block(embed_dim, drop_path=dpr[i], layer_idx=i, mixer_kwargs={"d_state": d_state, **mamba_kwargs}) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        x = self.norm(hidden_states)
        return x  # [batch, seq_len, embed_dim]

    def forward(self, x, **kwargs):
        return self.forward_features(x)

# --- Wrapper for SmolVLM ---
class VimBackbone(VisionMamba):
    def __init__(self, img_size=224, patch_size=16, stride=16, depth=4, embed_dim=768, channels=3, drop_path_rate=0.1, d_state=16, **mamba_kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, stride=stride, depth=depth, embed_dim=embed_dim, channels=channels, drop_path_rate=drop_path_rate, d_state=d_state, **mamba_kwargs)

    def forward(self, pixel_values, patch_attention_mask=None):
        features = super().forward(pixel_values)
        return type('Output', (), {'last_hidden_state': features})

# Now VimBackbone uses Mamba as the mixer in each block.
