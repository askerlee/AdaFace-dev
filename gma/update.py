import torch
import torch.nn as nn
import torch.nn.functional as F
from .gma import Aggregate

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, config):
        super(BasicMotionEncoder, self).__init__()
        # When corr_levels == 4, corr_radius == 4, num_corr_planes == 324.
        num_corr_planes = config.corr_levels * (2*config.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(num_corr_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv   = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, config, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.config = config
        self.encoder = BasicMotionEncoder(config)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net_feat, inp_feat, corr, flow):
        # motion_features: (256+2)-dimensional.
        motion_features = self.encoder(flow, corr)
        inp_feat = torch.cat([inp_feat, motion_features], dim=1)

        net_feat = self.gru(net_feat, inp_feat)
        delta_flow = self.flow_head(net_feat)

        # scale mask to balance gradients
        mask = .25 * self.mask(net_feat)
        return net_feat, mask, delta_flow


class GMAUpdateBlock(nn.Module):
    def __init__(self, config, hidden_dim=128):
        super().__init__()
        self.config = config
        self.encoder = BasicMotionEncoder(config)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        # Aggregate is attention with a (learnable-weighted) skip connection, without FFN.
        self.aggregator = Aggregate(config=self.config, dim=128, dim_head=128, heads=self.config.num_heads)

    # net_feat, inp_feat: [1, 128, 55, 128]. split from cnet_feat.
    def forward(self, net_feat, inp_feat, corr, flow, attention):
        # encoder: BasicMotionEncoder
        # corr: [3, 676, 50, 90]
        motion_features = self.encoder(flow, corr)
        # motion_features: 128-dim
        # attention: [8, 1, 2852, 2852]. motion_features: [8, 128, 46, 62].
        motion_features_global = self.aggregator(attention, motion_features)
    
        inp_cat = torch.cat([inp_feat, motion_features, motion_features_global], dim=1)

        # Attentional update
        net_feat = self.gru(net_feat, inp_cat)

        delta_flow = self.flow_head(net_feat)

        # scale mask to balence gradients
        mask = .25 * self.mask(net_feat)
        return net_feat, mask, delta_flow



