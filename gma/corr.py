import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.utils import bilinear_sampler
# from compute_sparse_correlation import compute_sparse_corr, compute_sparse_corr_torch, compute_sparse_corr_mink
import os

# CorrBlock has no learnable parameters.
class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, corr_normalized_by_sqrt_dim=True):
        self.num_levels = num_levels
        self.radius = radius
        
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2, corr_normalized_by_sqrt_dim)

        batch, h1, w1, dim, h2, w2 = corr.shape

        # The returned corr is like attention scores, not normalized.
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        # Save corr for visualization
        if 'SAVECORR' in os.environ:
            corr_savepath = os.environ['SAVECORR']
            corr2 = corr.detach().cpu().reshape(batch, h1, w1, h2, w2)
            torch.save(corr2, corr_savepath)
            print(f"Corr tensor saved to {corr_savepath}")

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # Sample the correlation map corr at the coordinates coords_lvl.
            # ** Essentially, this is do warping on the correlation map. **
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        # Concatenate the four levels (4 resolutions of neighbors), 
        # and permute the neighbors to the channel dimension.
        out = torch.cat(out_pyramid, dim=-1)
        # [batch, number of neighbors, h1, w1]
        return out.permute(0, 3, 1, 2).contiguous().float()

    # The returned corr is like attention scores, not softmax-normalized.
    @staticmethod
    def corr(fmap1, fmap2, corr_normalized_by_sqrt_dim=True):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        if corr_normalized_by_sqrt_dim:
            return corr / torch.sqrt(torch.tensor(dim).float())
        else:
            return corr

class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, do_corr_global_norm=False):
        super().__init__()
        self.radius = radius
        self.do_corr_global_norm = do_corr_global_norm
        
        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        if self.do_corr_global_norm:
            corr_3d = corr.permute(0, 3, 1, 2, 4, 5).view(B, dim, -1)
            corr_normed = F.layer_norm( corr_3d, (corr_3d.shape[2],), eps=1e-12 )
            corr = corr_normed.view(batch, dim, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
            
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.do_corr_global_norm = do_corr_global_norm
        
    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
