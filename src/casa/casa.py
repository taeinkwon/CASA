import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionalEncoding
from .casa_module import LocalFeatureTransformer
from .utils.matching import Matching
import tqdm


class CASA(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.ph = config['ph']['true']
        if self.ph:
            self.backbone_ph = build_backbone(config['ph'])
        if config['match']['d_model'] % 2 == 0:
            self.pos_encoding = PositionalEncoding(
                config['match']['d_model'])  # To make it even number
        else:
            self.pos_encoding = PositionalEncoding(
                config['match']['d_model']+1)  # To make it even number
        self.casa_coarse = LocalFeatureTransformer(config['match'])
        self.matching = Matching(config['match'])


    def forward(self, data, train=True): 
        """ 
        Update:
            data (dict): {
                'keypoints0': (torch.Tensor): (N, T, K, D) 1, 85, 25 3
                'keypoints1': (torch.Tensor): (N, T, K, D)
            }
        """
        if train:
            # For training set, we have two inputs, the original sequence and the 4D augmented sequence.

            # 1. Local Feature FCL
            data.update({
                'bs': data['keypoints0'].size(0),
                'hw0_i': data['keypoints0'].shape[1], 'hw1_i': data['keypoints1'].shape[1]
            })
            # else:  # handle different input shapes
            kp0 = torch.reshape(
                data['keypoints0'], (data['keypoints0'].shape[0], data['keypoints0'].shape[1], -1)).float()
            kp1 = torch.reshape(
                data['keypoints1'], (data['keypoints1'].shape[0], data['keypoints1'].shape[1], -1)).float()
            feat_f0, feat_f1 = self.backbone(kp0), self.backbone(kp1)

            data.update({
                # N T (K *D)
                'len_t0': feat_f0.shape[1], 'len_t1': feat_f1.shape[1], 'len_d': feat_f0.shape[2]
            })

            # 2. Matching
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            # positional encoding for CASA
            if self.config['match']['pe']:
                feat_f0 = self.pos_encoding(
                    feat_f0, data['steps0'], data['len0'])
                feat_f1 = self.pos_encoding(
                    feat_f1, data['steps1'], data['len1'])
            # CASA Matching
            feat_f0, feat_f1 = self.casa_coarse(
                feat_f0, feat_f1)

            data.update({
                # N T (K *D)
                'emb0': feat_f0, 'emb1': feat_f1
            })

            # 3. Projection Head
            # emb N T K*D
            if self.ph:
                z0 = self.backbone_ph(data['emb0'])
                z1 = self.backbone_ph(data['emb1'])
                data.update({
                    # N T (K *D)
                    'z0': z0,
                    'z1': z1
                })
                self.matching(z0, z1, data)
            else:
                self.matching(feat_f0, feat_f1, data)
        else:
            # For val set, we only need one input, the original sequence.
            data.update({
                'bs': data['keypoints0'].size(0),
                'hw0_i': data['keypoints0'].shape[1]
            })

            kp0 = torch.reshape(
                data['keypoints0'], (data['keypoints0'].shape[0], data['keypoints0'].shape[1], -1)).float()
            feat_f0 = self.backbone(kp0)
            data.update({
                # N T (K *D)
                'len_t0': feat_f0.shape[1], 'len_d': feat_f0.shape[2]
            })
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            if self.config['match']['pe']:
                feat_f0 = self.pos_encoding(
                    feat_f0, data['steps0'], data['len0'])
            # CASA Matching, we use the same sequence to get the features.
            feat_f0, feat_f1 = self.casa_coarse(
                feat_f0, feat_f0)
            # For val, we only use the latent space before projection head.
            data.update({
                # N T (K *D)
                'emb0': feat_f0
            })



    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
