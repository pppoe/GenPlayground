import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        args = t[:, None] * self.freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class PointFlowModel(nn.Module):
    """
    Transformer-based flow matching model for 2D point clouds.
    Each point is projected to a token; MHA lets all points attend to each other.
    No positional encoding — point clouds are unordered sets.

    Input:  x [B, N, 2]  noisy point cloud at time t
            t [B]        continuous time in [0, 1]
    Output:   [B, N, 2]  predicted velocity field
    """
    def __init__(self, hidden_dim=128, time_emb_dim=64, num_heads=4, depth=4):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.input_proj = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):
        # x: [B, N, 2], t: [B]
        tokens = self.input_proj(x) + self.time_proj(self.time_emb(t)).unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_proj(self.norm(tokens))  # [B, N, 2]


@torch.no_grad()
def sample(model, n, n_points=128, steps=10, device='cpu'):
    model.eval()
    x = torch.randn(n, n_points, 2, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), 1.0 - i * dt, device=device)  # 1 → 0 (denoising)
        x = x - model(x, t) * dt
    return x  # [n, n_points, 2]


if __name__ == '__main__':
    from render_tensor import render_image, make_grid

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PointFlowModel().to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))

    cols, rows, cell, gap = 4, 4, 128, 4
    pts = sample(model, cols * rows, device=device)          # [16, 128, 2]
    images = [render_image(p.cpu(), cell) for p in pts]
    grid = make_grid(cell, gap, cols, rows, images)
    grid.save('data/samples.jpg')
    print(f'saved data/samples.jpg  ({grid.width}x{grid.height})')
