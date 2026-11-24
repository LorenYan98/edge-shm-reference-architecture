"""Masked-autoencoding Vision Transformer for spectrogram images.

Dimensions follow the manuscript: 400 patch tokens (5x5 on a 100x100 image,
5x10 on a 100x200 image), encoder 3 layers / 768-d / 12 heads, decoder
4 layers / 512-d / 16 heads, 80 % masking during training, full (unmasked)
encode-decode at inference. Pure torch, no timm.
"""
import math
import torch
import torch.nn as nn

IMG, PATCH = (100, 100), (5, 5)      # defaults = v1 geometry


def sincos_2d(embed_dim, grid):
    """Fixed 2-D sine-cosine positional embedding, (gh*gw, embed_dim)."""
    def one_d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([out.sin(), out.cos()], dim=1)
    gh, gw = torch.meshgrid(torch.arange(grid[0], dtype=torch.float64),
                            torch.arange(grid[1], dtype=torch.float64), indexing='ij')
    emb = torch.cat([one_d(embed_dim // 2, gh.reshape(-1)), one_d(embed_dim // 2, gw.reshape(-1))], dim=1)
    return emb.float()


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class MAEViT(nn.Module):
    def __init__(self, embed_dim=768, depth=3, heads=12,
                 dec_dim=512, dec_depth=4, dec_heads=16, mask_ratio=0.8, img=IMG, patch=PATCH):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img, self.psz = tuple(img), tuple(patch)
        grid = (self.img[0] // self.psz[0], self.img[1] // self.psz[1])
        self.grid = grid; self.npatch = grid[0] * grid[1]
        self.patch = nn.Conv2d(1, embed_dim, self.psz, self.psz)
        self.register_buffer('pos', sincos_2d(embed_dim, grid).unsqueeze(0))
        self.blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.dec_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer('dec_pos', sincos_2d(dec_dim, grid).unsqueeze(0))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, self.psz[0] * self.psz[1])
        nn.init.normal_(self.mask_token, std=0.02)

    # -- patch helpers
    def patchify(self, img):                # (B,1,H,W) -> (B,npatch,ph*pw)
        B = img.shape[0]; (gh, gw), (ph, pw) = self.grid, self.psz
        x = img.reshape(B, 1, gh, ph, gw, pw).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, gh * gw, ph * pw)

    def unpatchify(self, x):                # inverse of patchify
        B = x.shape[0]; (gh, gw), (ph, pw) = self.grid, self.psz
        x = x.reshape(B, gh, gw, ph, pw, 1).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, 1, self.img[0], self.img[1])

    def random_mask(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        shuffle = noise.argsort(dim=1)
        restore = shuffle.argsort(dim=1)
        kept = shuffle[:, :keep]
        x_vis = torch.gather(x, 1, kept.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, restore)   # 1 = masked
        return x_vis, mask, restore

    def encode(self, img, masked):
        x = self.patch(img).flatten(2).transpose(1, 2) + self.pos
        if masked:
            x, mask, restore = self.random_mask(x)
        else:
            mask = torch.zeros(x.shape[0], self.npatch, device=x.device); restore = None
        for b in self.blocks:
            x = b(x)
        return self.norm(x), mask, restore

    def decode(self, z, restore):
        z = self.dec_embed(z)
        if restore is not None:
            B, keep, D = z.shape
            toks = self.mask_token.expand(B, self.npatch - keep, D)
            z = torch.cat([z, toks], dim=1)
            z = torch.gather(z, 1, restore.unsqueeze(-1).expand(-1, -1, D))
        z = z + self.dec_pos
        for b in self.dec_blocks:
            z = b(z)
        return self.pred(self.dec_norm(z))

    def forward(self, img, masked=True):
        """Returns (loss, pred_patches, mask). Training loss is MSE on masked
        patches; with masked=False the loss is the full-image MSE used as the
        anomaly score."""
        z, mask, restore = self.encode(img, masked)
        pred = self.decode(z, restore)
        target = self.patchify(img)
        per_patch = ((pred - target) ** 2).mean(dim=-1)
        if masked:
            loss = (per_patch * mask).sum() / mask.sum()
        else:
            loss = per_patch.mean()
        return loss, pred, mask

    @torch.no_grad()
    def score(self, img):
        """Per-sample reconstruction error e = ||X - X_hat||^2 / N, unmasked."""
        z, _, _ = self.encode(img, masked=False)
        pred = self.decode(z, None)
        return ((pred - self.patchify(img)) ** 2).mean(dim=(1, 2))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    m = MAEViT()
    enc = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(('dec_', 'pred', 'mask_token')))
    print('params total %.1f M, encoder %.1f M, decoder %.1f M' % (n_params(m) / 1e6, enc / 1e6, (n_params(m) - enc) / 1e6))
    for img, patch in ((IMG, PATCH), ((100, 200), (5, 10))):
        m = MAEViT(img=img, patch=patch)
        x = torch.rand(2, 1, *img)
        loss, pred, mask = m(x)
        print(img, patch, 'tokens', m.npatch, 'params %.1f M' % (n_params(m) / 1e6),
              'train loss %.3f' % float(loss), 'mask frac', float(mask.mean()), 'score', [round(v, 4) for v in m.score(x).tolist()])
