from numpy.lib.twodim_base import mask_indices
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class CASALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['casa']['loss']
        self.match_type = self.config['casa']['match']['match_type']
        self.match_algo = self.config['casa']['match']['match_algo']
        self.mse_loss = nn.MSELoss()
        # coarse-level
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        self.use_prior = self.config['casa']['match']['use_prior']
        self.DEBUG = False

    def compute_coarse_loss(self, data, weight=None):
        steps0 = data['steps0']
        steps1 = data['steps1']

        conf = data['conf_matrix']
        conf_prior = data['conf_matrix_prior']
        i_mask = data['i_mask']
        j_mask = data['j_mask']
        gt_mask_float = (i_mask.float()+j_mask.float())/2.0
        gt_mask = i_mask + j_mask

        if self.match_type == 'dual_softmax':

            conf0 = data['conf_matrix0']
            conf1 = data['conf_matrix1']
            len0 = data['len0']
            len1 = data['len1']
            sim_matrix = data['sim_matrix']


            neg_mask = gt_mask == False
            mask = i_mask * j_mask

        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w

        # mask = (conf_prior ==1)

        if self.loss_config['type'] == 'cross_entropy':

            if self.match_type == 'dual_softmax':
                conf = torch.clamp(conf, 1e-6, 1-1e-6)

                if self.use_prior:
                    conf_prior = conf_prior.bool()
                    gt_mask = conf_prior

                loss_pos = []
                if self.loss_config['loss_type'] == 'regression':
                    # cal row and col seperately

                    steps_i = torch.unsqueeze(steps0, 2)

                    true_time_i = torch.sum(steps_i*gt_mask.float(), 1)
                    pred_time_i = torch.sum(steps_i*conf0, 1)
                    gt_mask_sum_i = torch.sum(gt_mask.float(), dim=1) == 1
                    loss_pos = self.mse_loss(
                        true_time_i[gt_mask_sum_i], pred_time_i[gt_mask_sum_i])

                elif self.loss_config['loss_type'] == 'regression_var':

                    coeff_lambda = 0.00001
                    num_list_i = torch.unsqueeze(steps0, 2)
                    conf_num_i = conf0 * num_list_i
                    mean_conf_i = torch.sum(conf_num_i, 1)
                    diff_i = num_list_i - torch.unsqueeze(mean_conf_i, 1)
                    sigma_square_i = torch.sum(conf0 * diff_i**2, 1)
                    mask_loc_i = torch.argmax(gt_mask.float(), dim=1)
                    gt_mask_sum_i = torch.sum(gt_mask.float(), dim=1) == 1
                    diff_regression_i = torch.gather(
                        steps0, 1, mask_loc_i) - mean_conf_i

                    num_list_j = torch.unsqueeze(steps1, 1)
                    conf_num_j = conf1 * num_list_j
                    mean_conf_j = torch.sum(conf_num_j, 2)
                    diff_j = num_list_j - torch.unsqueeze(mean_conf_j, 2)
                    sigma_square_j = torch.sum(conf1 * diff_j**2, 2)
                    mask_loc_j = torch.argmax(gt_mask.float(), dim=2)
                    gt_mask_sum_j = torch.sum(gt_mask.float(), dim=2) == 1
                    diff_regression_j = torch.gather(
                        steps1, 1, mask_loc_j) - mean_conf_j

                    if self.DEBUG:  # debug purpose
                        print("gt_mask_sum", gt_mask_sum_j.shape)
                        print("gt_mask[0]", gt_mask[0])
                        print("mask_loc_i[0]", mask_loc_j[0])
                        print("num_list_i", num_list_j.shape)
                        print("conf0", conf1.shape)
                        print("conf_num_i", conf_num_j.shape)
                        print("num_list_i", num_list_j.shape)
                        print("torch.unsqueeze(mean_conf_i,1)",
                              torch.unsqueeze(mean_conf_j, 2).shape)
                        print("diff_i", diff_j.shape)
                        print("num_list_i", num_list_j.shape)
                        print("conf0", conf1.shape)
                        print("diff_i", diff_j.shape)
                        print("mask_loc_i", mask_loc_j.shape)
                        print("mean_conf_i", mean_conf_j.shape)
                        print("sigma_square_i", sigma_square_j.shape)
                        print("diff_regression_i", diff_regression_j.shape)
                        print("(diff_regression_i[gt_mask_sum]**2)/sigma_square_i[gt_mask_sum]", ((
                            diff_regression_j[gt_mask_sum_j]**2)/sigma_square_j[gt_mask_sum_j]).shape)
                    # loss_pos += conf0[gt_mask_sum]
                    loss_pos = (diff_regression_i[gt_mask_sum_i]**2)/sigma_square_i[gt_mask_sum_i] + \
                        coeff_lambda * torch.log(sigma_square_i[gt_mask_sum_i])
                    loss_pos += (diff_regression_j[gt_mask_sum_j]**2)/sigma_square_j[gt_mask_sum_j] + \
                        coeff_lambda * torch.log(sigma_square_j[gt_mask_sum_j])
                else:
                    raise ValueError('Supported loss types: regression and '
                                     'regression_var.')

                return loss_pos.mean()  # + loss_count.mean()

        elif self.loss_config['type'] == 'mse':
            loss = self.mse_loss(conf, gt_mask_float)
            return loss  # .mean()
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(
                type=self.loss_config['type']))

    @ torch.no_grad()
    def compute_c_weight(self, data):
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None]
                        * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        loss_scalars = {}
        # compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # computer loss loss
        loss_c = self.compute_coarse_loss(data,
                                          weight=c_weight)

        loss = loss_c * self.loss_config['weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
