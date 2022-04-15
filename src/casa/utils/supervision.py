from math import log
from loguru import logger

import torch
from einops import repeat
#from kornia.utils import create_meshgrid
import numpy as np
import scipy.stats as stats
import math

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
    """
    # 1. misc
    device = data['keypoints0'].device
    N0, T0, K0, D0 = data['keypoints0'].shape
    N1, T1, K1, D1 = data['keypoints1'].shape
    max_len_t0 = data['len0']
    max_len_t1 = data['len1']

    #print("N0, T0, K0, D0", N0, T0, K0, D0)
    # Gaussian prior matrix
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    #x = np.linspace(mu - 2*sigma, mu + 2*sigma, T0*2)
    #y = stats.norm.pdf(x, mu, sigma)
    if config.CONSTRASTIVE.TRAIN:
        conf_matrix_prior = torch.zeros(N0, T0, T1, device=device)
        for nn in range(N0):
            for ii in range(max_len_t0[nn]):
                if data['matching'][nn][ii] == -1:
                    continue
                jj_index = data['matching'][nn][ii]
                conf_matrix_prior[nn][ii][jj_index] = 1

    else:
        conf_matrix_prior = torch.ones(T0, T1, device=device)
        for ii in range(T0):
            for jj in range(T1):
                # Put Gaussian dtribution
                #conf_matrix_prior[ii][jj] = y[T0+abs(ii-jj)]
                # Or, just diagonal matrix
                if (ii != jj):  # and abs(ii-jj) != 1:
                    conf_matrix_prior[ii][jj] = 0
        conf_matrix_prior = conf_matrix_prior.repeat(N0, 1, 1)

    data.update({'conf_matrix_prior': conf_matrix_prior})


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])
               ) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['pennaction', 'h2o', 'ikea_asm']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')

