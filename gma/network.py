import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils.utils import coords_grid, upflow8, print0
from .gma import Attention
from easydict import EasyDict as edict

class GMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config is None:
            self.config = edict()
        else:
            self.config = edict(config)

        self.hidden_dim  = hdim = 128
        self.context_dim = cdim = 128

        # corr_levels determines the shape of the model params. 
        # So it cannot be changed arbitrarily.
        if not hasattr(self.config, 'corr_levels'):
            self.config.corr_levels = 4
        if not hasattr(self.config, 'corr_radius'):
            self.config.corr_radius = 4
        if not hasattr(self.config, 'dropout'):
            self.config.dropout = 0
        if not hasattr(self.config, 'mixed_precision'):
            self.config.mixed_precision = True
        if not hasattr(self.config, 'num_heads'):
            self.config.num_heads = 1
        if not hasattr(self.config, 'corr_normalized_by_sqrt_dim'):
            self.config.corr_normalized_by_sqrt_dim = True

        print0(f"corr_levels: {self.config.corr_levels}, corr_radius: {self.config.corr_radius}")
                
        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256,         norm_fn='instance', dropout=self.config.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch',    dropout=self.config.dropout)
        self.att  = Attention(config=self.config, dim=cdim, heads=self.config.num_heads, 
                              max_pos_size=160, dim_head=cdim)

        # GMAUpdateBlock() accesses corr_levels, corr_radius and num_heads from config.
        self.update_block = GMAUpdateBlock(self.config, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, num_iters=12, flow_init=None, upsample=True, test_mode=0):
        """ Estimate optical flow between a pair of frames """

        # image1, image2: [1, 3, 440, 1024]
        # image1 mean: [-0.1528, -0.2493, -0.3334]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        # fmap1, fmap2: [1, 256, 55, 128]. 1/8 size of the original image.
        # correlation matrix: 7040*7040 (55*128=7040).
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
            
        self.corr_fn = CorrBlock(fmap1, fmap2, radius=self.config.corr_radius)

        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            # run the context network
            # cnet: context network to extract features from image1 only.
            # cnet arch is the same as fnet. 
            # fnet extracts features specifically for correlation computation.
            # cnet_feat: extracted features focus on semantics of image1? 
            # (semantics of each pixel, used to guess its motion?)
            cnet_feat = self.cnet(image1)

            # Both fnet and cnet are BasicEncoder. output is from conv (no activation function yet).
            # net_feat, inp_feat: [1, 128, 55, 128]
            net_feat, inp_feat = torch.split(cnet_feat, [hdim, cdim], dim=1)
            net_feat = torch.tanh(net_feat)
            inp_feat = torch.relu(inp_feat)

            # attention, att_c, att_p = self.att(inp_feat)
            # self.att: Intra-frame attention prob matrix. We've set out_attn_probs_only=True.
            attention = self.att(inp_feat)
                
        # coords0 is always fixed as original coords.
        # coords1 is iteratively updated as coords0 + current estimated flow.
        # At this moment coords0 == coords1.
        coords0, coords1 = self.initialize_flow(image1)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(num_iters):
            coords1 = coords1.detach()
            # corr: [6, 324, 50, 90]. 324: number of points in the neighborhood. 
            # radius = 4 -> neighbor points = (4*2+1)^2 = 81. 4 levels: x4 -> 324.
            corr = self.corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                # net_feat: hidden features of SepConvGRU. 
                # inp_feat: input  features to SepConvGRU.
                # up_mask is scaled to 0.25 of original values.
                # update_block: GMAUpdateBlock
                # In the first few iterations, delta_flow.abs().max() could be 1.3 or 0.8. Later it becomes 0.2~0.3.
                # attention is the intra-frame attention matrix of inp_feat. 
                # It's only used to aggregate motion_features and form motion_features_global.
                net_feat, up_mask, delta_flow = self.update_block(net_feat, inp_feat, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                # coords0 is fixed as original coords.
                # upflow8: upsize to 8 * height, 8 * width. 
                # flow value also *8 (scale the offsets proportionally to the resolution).
                flow_up = upflow8(coords1 - coords0)
            else:
                # The final high resolution flow field is found 
                # by using the mask to take a weighted combination over the neighborhood.
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode == 1:
            return coords1 - coords0, flow_up
        if test_mode == 2:
            return coords1 - coords0, flow_predictions
            
        return flow_predictions

    def est_flow_from_feats(self, fmap1, fmap2, H, W, num_iters=3, flow_init=None):
        """ Estimate optical flow between a pair of frame features """
        hdim = self.hidden_dim
        cdim = self.context_dim
        BS   = fmap1.shape[0]
        # If fmap1 and fmap2 are from attention layers, H and W are collapsed.
        # So we need to restore the 4D shape.
        fmap1 = fmap1.reshape(BS, -1, H, W)
        fmap2 = fmap2.reshape(BS, -1, H, W)

        H0, W0 = H, W
        # After corr_levels -1 = 3 levels of pooling, the feature map size is 1/8 of the original image.
        # So the minimum size of the feature map is 1/8 of the original image.
        # If the minimum size is less than 16, the top level of the correlation pyramid will be 1x1.
        # In this case, we enlarge the feature map by 4x to make the top level of the pyramid 4x4.
        if min(H, W) < 16:
            fmap1 = F.interpolate(fmap1, scale_factor=4, mode='bilinear', align_corners=False)
            fmap2 = F.interpolate(fmap2, scale_factor=4, mode='bilinear', align_corners=False)
            H, W = H * 4, W * 4
            scale_factor = 4
        else:
            scale_factor = 1

        # CorrBlock has no learnable parameters.
        self.corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.config.corr_levels, 
                                 radius=self.config.corr_radius, 
                                 corr_normalized_by_sqrt_dim=self.config.corr_normalized_by_sqrt_dim)

        net_feat = torch.zeros(BS, hdim, H, W).to(fmap1.device)
        inp_feat = torch.zeros(BS, cdim, H, W).to(fmap1.device)

        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            # attention is a uniform matrix, since inp_feat is all zeros.
            attention = self.att(inp_feat)
                
        # coords0 is always fixed as original coords.
        # coords1 is iteratively updated as coords0 + current estimated flow.
        # At this moment coords0 == coords1.
        # inp_feat: [BS, 128, H, W]. Only H and W are used to initialize coords0 and coords1.
        # So passing inp_feat is sufficient.
        coords0, coords1 = self.initialize_flow(inp_feat)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(num_iters):
            coords1 = coords1.detach()
            # Warp correlation volume with the current flow estimate, 
            # to get the correlation volume at the next iteration.
            # corr: [6, 324, 50, 90]. 324: number of points in the neighborhood. 
            # radius = 4 -> neighbor points = (4*2+1)^2 = 81. 4 levels: x4 -> 324.
            corr = self.corr_fn(coords1)  
            flow = coords1 - coords0
            
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                # net_feat: hidden features of SepConvGRU. 
                # inp_feat: input  features to SepConvGRU.
                # up_mask is scaled to 0.25 of original values.
                # update_block: GMAUpdateBlock
                # In the first few iterations, delta_flow.abs().max() could be 1.3 or 0.8. Later it becomes 0.2~0.3.
                # attention is the intra-frame attention matrix of inp_feat. 
                # It's only used to aggregate motion_features and form motion_features_global.
                net_feat, up_mask, delta_flow = self.update_block(net_feat, inp_feat, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow = coords1 - coords0

            flow_predictions.append(flow)
    
        final_flow = coords1 - coords0
        if scale_factor > 1:
            final_flow = F.interpolate(final_flow, size=(H0, W0), mode='bilinear', align_corners=False)

        # Collapse H, W dimensions.
        final_flow = final_flow.reshape(BS, 2, H0*W0)
        return final_flow
