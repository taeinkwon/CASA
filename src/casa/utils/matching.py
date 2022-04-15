import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

INF = 1e9


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        m (torch.Tensor): [N, L0, L1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    # m[:, :, :, :b] = v
    # m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    # m[:, :, :, -b:] = v
    # m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded maskszzz
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def pairwise_l2_distance(embs1, embs2):
    """Computes pairwise distances between all rows of embs1 and embs2."""
    n, l, c = embs1.shape
    norm1 = torch.sum(torch.square(embs1), 2)
    norm1 = torch.reshape(norm1, [n, -1, 1])
    norm2 = torch.sum(torch.square(embs2), 2)
    norm2 = torch.reshape(norm2, [n, 1, -1])

    # Max to ensure matmul doesn't produce anything negative due to floating
    # point approximations.
    dist = torch.maximum(
        norm1 + norm2 - 2.0 * torch.einsum("nlc,nsc->nls", embs1, embs2), torch.zeros_like(norm1))

    return dist


def get_scaled_similarity(embs1, embs2, temperature):

    B, M, C = embs1.shape
    # Go for embs1 to embs2.
    similarity = -pairwise_l2_distance(embs1, embs2)
    similarity /= C
    similarity /= temperature

    return similarity


def dual_softmax(feat_c0, feat_c1, temperature, SIM=False):
    # print("feat_c0", feat_c0)
    N, L, S, C = feat_c0.size(0), feat_c0.size(
        1), feat_c1.size(1), feat_c0.size(2)

    # normalize
    feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                           [feat_c0, feat_c1])

    if SIM:
        # feat_c0 = F.normalize(
        #    feat_c0, p=2.0, dim=1, eps=1e-12, out=None)
        # feat_c1 = F.normalize(
        #    feat_c1, p=2.0, dim=1, eps=1e-12, out=None)
        #print("feat_c0", feat_c0.shape)
        norm_c0 = torch.unsqueeze(torch.norm(feat_c0, dim=2), 2)
        norm_c1 = torch.unsqueeze(torch.norm(feat_c1, dim=2), 2)
        #print("norm_c0", norm_c0.shape)
        feat_c0 = feat_c0 / norm_c0
        feat_c1 = feat_c1/norm_c1
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                  feat_c1) / temperature
    else:  # distnace matrix
        # Temperature helps with how soft the alignment should be.
        sim_matrix = -torch.cdist(feat_c0, feat_c1)
        sim_matrix /= temperature
        # Scale the distance  by number of channels. This normalization helps with optimization.
        sim_matrix /= C
        # print("sim_matrix",sim_matrix.shape)
        # = torch.einsum("nlc,nsc->nls", feat_c0,
        #                        feat_c1) / temperature
    conf_matrix0 = F.softmax(sim_matrix, 1)
    conf_matrix1 = F.softmax(sim_matrix, 2)

    conf_matrix = conf_matrix0 * conf_matrix1
    return conf_matrix, conf_matrix0, conf_matrix1, sim_matrix



def dual_bicross(feat_c0, feat_c1):
    # print("feat_c0", feat_c0)
    N, L, S, C = feat_c0.size(0), feat_c0.size(
        1), feat_c1.size(1), feat_c0.size(2)

    # normalize
    feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                           [feat_c0, feat_c1])
    sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                              feat_c1)
    sigmoid_f = nn.Sigmoid()
    conf_matrix = sigmoid_f(sim_matrix)
    return conf_matrix


class Matching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        self.temperature = config['dsmax_temperature']
        self.sim = config['similarity']


    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        if self.match_type == 'dual_softmax':
            conf_matrix, conf_matrix0, conf_matrix1, sim_matrix = dual_softmax(
                feat_c0, feat_c1, self.temperature, SIM=self.sim)
            data.update({'conf_matrix': conf_matrix, 'conf_matrix0': conf_matrix0,
                        'conf_matrix1': conf_matrix1, 'sim_matrix': sim_matrix})
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match(conf_matrix, data))
        elif self.match_type == 'dual_bicross':
            conf_matrix = dual_bicross(feat_c0, feat_c1)
            data.update({'conf_matrix': conf_matrix})
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match(conf_matrix, data))


    @ torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'len0': data['len_t0'],
            # 'w0c': data['hw0_c'][1],
            'len1': data['len_t1'],
            # 'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr

        # 2. mutual nearest
        i_mask = conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]
        j_mask = conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]
        mask = mask * i_mask * j_mask

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'mask': mask,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mconf': mconf[mconf != 0],
            'i_mask': i_mask,
            'j_mask': j_mask,
        })

        return coarse_matches
