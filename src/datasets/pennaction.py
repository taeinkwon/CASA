from os import path as osp
from typing import Dict
from unicodedata import name
import itertools
import copy
from numpy.lib.function_base import insert
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
import random
from dataset_preparation.preprocess_norm_mat import pre_normalization_mat, get_openpose_connectivity
from bodymocap.models import SMPLX
import time

# Loading VPoser Body Pose Prior
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

# Mano
from manopth.manolayer import ManoLayer

# Drawing tools.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import subprocess

from dataset_splits import DATASETS


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


class PennActionDataset(utils.data.Dataset):
    def __init__(self,
                 npz_path,
                 num_frames=None,
                 sampling_strategy=None,
                 augmentation_strategy=None,
                 mode='train',
                 augment_fn=None,
                 pose_dir=None,
                 contrastive=False,
                 val=False,
                 use_norm=True,
                 config=None,
                 **kwargs):

        super().__init__()
        # self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.config = config
        self.mode = mode
        self.contrastive = contrastive
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.use_norm = use_norm
        self.smpl = self.config.DATASET.SMPL
        self.mano = self.config.DATASET.MANO
        self.augmentation_strategy = augmentation_strategy
        self.val = val
        self.max_len = DATASETS[self.config.DATASET.NAME]['max_len']
        self.dataset_path = self.config.DATASET.PATH

        # prepare data_names, intrinsics and extrinsics(T)
        print("npz_path", npz_path)
        self.mode_str = npz_path.split('.')[-2].split('_')[-1]
        npz_path = os.path.join(self.dataset_path, npz_path)
        if not self.use_norm:
            npz_path = npz_path + 'nn'
        elif self.smpl:
            npz_path = npz_path  # + 'nn2'
            smpl_dir = os.path.join(self.dataset_path, 'smpl/models')
            print("self.dataset_path", self.dataset_path)
            print("smpl_dir", smpl_dir)
            self.smpl_model = SMPLX(smpl_dir,
                                    batch_size=self.max_len,
                                    num_betas=10,
                                    use_pca=False,
                                    create_transl=False)

            self.device = torch.device('cuda')  # cpu cuda
            self.smpl_model.to(device=self.device)
            self.smpl_keypoints = {}

            support_dir = os.path.join(
                self.dataset_path, "human_body_prior/support_data/downloads/")
            expr_dir = osp.join(support_dir, 'vposer_v2_05')
            vp, ps = load_model(expr_dir, model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True)
            self.vp = vp.to(device=self.device)
        elif self.mano:
            ncomps = 45
            mano_dir = os.path.join(self.dataset_path, 'mano/models')
            self.mano_layer_l = ManoLayer(
                mano_root=mano_dir, use_pca=False, ncomps=ncomps, flat_hand_mean=True, side='left')
            self.mano_layer_r = ManoLayer(
                mano_root=mano_dir, use_pca=False, ncomps=ncomps, flat_hand_mean=True, side='right')

            self.device = torch.device('cuda')  # cpu cuda
            self.mano_layer_l.to(device=self.device)
            self.mano_layer_r.to(device=self.device)

            self.mano_keypoints = {}

        data = np.load(npz_path, allow_pickle=True)
        self.data = data.item()



        if self.val:
            self.data_names = list(self.data.keys())
        else:
            self.data_names = list(self.data.keys()) * 50

        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def get_keypoints_from_smpl(self, pose, beta):

        M, D1 = beta.shape
        M, D2 = pose.shape
        betas = torch.zeros(self.max_len, D1)
        body_pose = torch.zeros(self.max_len, D2)
        betas[:M, :] = torch.from_numpy(beta.astype(np.float32))
        body_pose[:M, :] = torch.from_numpy(pose.astype(np.float32))

        body_pose = body_pose.to(device=self.device)
        betas = betas.to(device=self.device)
        smpl_output = self.smpl_model(
            global_orient=body_pose[:, :3],
            body_pose=body_pose[:, 3:],
            betas=betas)
        #middle_t = time.time()

        keypoints = pre_normalization_mat(np.reshape(smpl_output.joints.detach().cpu().numpy()[
            :M, :25], (-1, 25, 3)), zaxis=[1, 8], xaxis=[1, 5])
        return keypoints

    def get_keypoints_from_mano(self, pose, beta):

        M, D1 = beta.shape
        M, D2 = pose.shape
        betas = torch.zeros(self.max_len, D1)
        mano_pose = torch.zeros(self.max_len, D2)
        betas[:M, :] = torch.from_numpy(beta.astype(np.float32))
        mano_pose[:M, :] = torch.from_numpy(pose.astype(np.float32))

        mano_pose = mano_pose.to(device=self.device)
        betas = betas.to(device=self.device)

        trans_l = torch.unsqueeze(mano_pose[:, :3], 1)
        trans_r = torch.unsqueeze(mano_pose[:, 51:54], 1)

        _, mano_output_l = self.mano_layer_l(mano_pose[:, 3:51], betas[:, :10])
        _, mano_output_r = self.mano_layer_l(mano_pose[:, 54:], betas[:, 10:])


        mano_output = torch.cat(
            (mano_output_l+trans_l, mano_output_r+trans_r), axis=1)
        keypoints = pre_normalization_mat(np.reshape(mano_output.detach().cpu().numpy()[
            :M, :42], (-1, 42, 3)), zaxis=[0, 1], xaxis=[0, 9])
        return keypoints

  
    def sample_steps(self, data_name):
        def _sample_uniform(item_len):
            interval = item_len // self.num_frames
            steps = range(0, item_len, interval)[:self.num_frames]
            return sorted(steps)

        def _sample_random(item_len):
            steps = random.sample(
                range(1, item_len), self.num_frames)
            return sorted(steps)

        def _sample_all():
            return list(range(0, self.num_frames))

        len0 = len(self.data[data_name[0]]['labels'])
        check0 = (self.num_frames <= len0)
        if check0:
            steps0 = _sample_random(len0)
        else:
            steps0 = _sample_all()
        len1 = len(self.data[data_name[1]]['labels'])
        check1 = (self.num_frames <= len1)
        if check1:
            steps1 = _sample_random(len1)
        else:
            steps1 = _sample_all()

        return torch.IntTensor(steps0), torch.IntTensor(steps1)

    def sample_steps_one(self, len0):
        # num_frames=20
        def _sample_uniform(item_len):
            interval = item_len // self.num_frames
            steps = range(0, item_len, interval)[:self.num_frames]
            return sorted(steps)

        def _sample_random(item_len):
            steps = random.sample(
                range(1, item_len), self.num_frames)
            return sorted(steps)

        def _sample_all():
            return list(range(0, self.num_frames))

        check0 = (self.num_frames <= len0)
        if check0:
            steps0 = _sample_random(len0)
        else:
            steps0 = _sample_all()

        return torch.IntTensor(steps0)

    def get_steps(self, step):
        """Sample multiple context steps for a given step."""

        num_steps = self.config.DATASET.NUM_STEPS
        stride = self.config.DATASET.FRAME_STRIDE

        if num_steps < 1:
            return step
        if stride < 1:
            raise ValueError('stride should be >= 1.')
        steps = torch.arange(step - (num_steps - 1) *
                             stride, step + stride, stride)

        return steps

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        

        if self.contrastive:

            len_st = len(self.augmentation_strategy)
            strategy = self.augmentation_strategy

            len0 = len(self.data[data_name]['labels'])
            len1 = len0
            vposer_prob0 = 1
            # Todo: Get the value directly from the setting.
            dim, channel = self.config.CASA.MATCH.D_MODEL//3, 3  # 25, 3

            if self.config.DATASET.ATT_STYLE:

                # print("hihi")
                NO_TIME_AUG = ('fast' not in strategy) and (
                    'slow' not in strategy)
                steps0 = np.array(list(range(self.max_len)))
                steps1 = np.array(list(range(self.max_len)))

                keypoints0 = np.zeros([self.max_len, dim, channel])
                keypoints1 = np.zeros([self.max_len, dim, channel])

                label0 = np.ones([self.max_len]) * (self.max_len-1)
                label1 = np.ones([self.max_len]) * (self.max_len-1)

                # if NO_TIME_AUG:
                #steps0 = self.sample_steps_one(len0)
                matched_list = list(range(self.max_len))
                if self.smpl:
                    T, D = self.data[data_name]['pose'].shape
                    if data_name in self.smpl_keypoints.keys():
                        keypoints0[:len0, :,
                                   :] = self.smpl_keypoints[data_name]
                    else:
                        keypoints0[:len0, :, :] = self.get_keypoints_from_smpl(
                            self.data[data_name]['pose'], self.data[data_name]['beta'])
                        self.smpl_keypoints[data_name] = keypoints0[:len0, :, :]
                elif self.mano:
                    T, D = self.data[data_name]['pose'].shape
                    if data_name in self.mano_keypoints.keys():
                        keypoints0[:len0, :,
                                   :] = self.mano_keypoints[data_name]
                    else:
                        keypoints0[:len0, :, :] = self.get_keypoints_from_mano(
                            self.data[data_name]['pose'], self.data[data_name]['beta'])
                        self.mano_keypoints[data_name] = keypoints0[:len0, :, :]
                else:
                    keypoints0[:len0, :, :] = self.data[data_name]['pose']
                label0[:len0] = self.data[data_name]['labels']

                keypoints1 = copy.deepcopy(keypoints0)
                label1 = copy.deepcopy(label0)
                steps1 = copy.deepcopy(steps0)

                steps1 = np.array(steps1)
                steps0 = np.array(steps0)

                if self.val:
                    # If it is val, need to put back the same seq as the original
                    # No 4D augmentation.
                    data = {
                        'len0': len0,
                        'len1': len1,
                        'steps0': steps0,
                        'steps1': steps1,
                        'mode': self.mode_str,
                        'keypoints0': keypoints0,   # (1, h, w)
                        'label0': label0,   # (h, w)
                        'keypoints1': keypoints1,
                        'label1': label1,
                        'matching': torch.IntTensor(matched_list),
                        'dataset_name': 'PennAction',
                        'pair_id': idx,
                        'pair_names': data_name
                    }
                    return data

                # time augmentation before translation noise and flipping.
                slow_fast_prob = np.random.uniform(low=-0, high=1)
                if 'fast' in strategy:
                    if ('slow' not in strategy) or slow_fast_prob > 0.5:
                        fast_coeff = np.random.uniform(low=0, high=0.5)
                        steps0_eff = np.array(list(range(len0)))
                        steps1_eff = np.array(list(range(len1)))
                        sampled_steps = random.sample(
                            range(len0), int(fast_coeff*len0))
                        steps1_eff = np.delete(steps1_eff, sampled_steps)
                        len1 = len(steps1_eff)

                        # Initialize keypoints1 and lable 1
                        keypoints1 = np.zeros([self.max_len, dim, channel])
                        label1 = np.ones([self.max_len]) * (self.max_len-1)

                        # Assign numbers to keypoints1 and labels based on steps1 we calculated above.
                        keypoints1[:len1, :, :] = keypoints0[steps1_eff]
                        label1[:len1] = label0[steps1_eff]

                        #len_new_label0 = len(label0)
                        matched_list = [-1] * self.max_len

                        nn0 = [np.argmin(abs(steps1_eff - b))
                               for b in steps0_eff]
                        nn1 = [np.argmin(abs(steps0_eff - b))
                               for b in steps1_eff]

                        if self.smpl or self.mano:
                            # for the geometric augmentation
                            pose_values = self.data[data_name]['pose'][steps1_eff]
                            beta_values = self.data[data_name]['beta'][steps1_eff]

                        for ii in range(len0):
                            if ii == nn1[nn0[ii]]:
                                matched_list[ii] = nn0[ii]

                if 'slow' in strategy:
                    if ('fast' not in strategy) or slow_fast_prob < 0.5:
                        def cal_interpolation(pose1, sampled_steps, len1):
                            pose_vals = []
                            for step in sampled_steps:
                                if step == len1-1:
                                    pose_val = pose1[step]
                                else:
                                    # print("pose_val",pose_val.shape)
                                    euler_angle0 = R.from_rotvec(np.reshape(
                                        pose1[step], (-1, 3))).as_euler('zyx', degrees=True)
                                    #M, K = euler_angle.shape
                                    euler_angle1 = R.from_rotvec(np.reshape(
                                        pose1[step+1], (-1, 3))).as_euler('zyx', degrees=True)

                                    euler_angle = (
                                        euler_angle0 + euler_angle1)/2
                                    pose_val = np.reshape(R.from_euler(
                                        'zyx', euler_angle, degrees=True).as_rotvec(), (-1))
                                pose_vals.append(pose_val)

                            return np.array(pose_vals)

                        slow_coeff = np.random.uniform(low=0, high=0.5)
                        steps0_eff = np.array(list(range(len0)))
                        steps1_eff = np.array(list(range(len1)))

                        # Select duplicated frames.
                        select_num = self.max_len - len0
                        sampled_steps = np.array(random.sample(
                            range(len0), min(int(slow_coeff*len0), select_num)))

                        if len(sampled_steps) > 0:
                            # Initialize keypoints1, pose1 and lable 1
                            keypoints1 = np.zeros([self.max_len, dim, channel])
                            pose_values = np.zeros([self.max_len, D])
                            label1 = np.ones([self.max_len]) * (self.max_len-1)

                            # Define step1
                            steps1_eff = np.concatenate(
                                (steps1_eff, sampled_steps), axis=0)
                            steps1_eff = np.sort(steps1_eff)
                            # Assign numbers to keypoints1 and labels based on steps1 we calculated above.
                            interpolated_pose = cal_interpolation(
                                self.data[data_name]['pose'], sampled_steps, len1)

                            len1 = len(steps1_eff)

                            # Assign numbers to keypoints1 and labels based on steps1 we calculated above.
                            pose_values = np.insert(
                                self.data[data_name]['pose'], sampled_steps+1, interpolated_pose, axis=0)[:len1]

                            beta_values = self.data[data_name]['beta'][steps1_eff]
                            label1[:len1] = label0[steps1_eff]

                            keypoints1[:len1, :, :] = self.get_keypoints_from_smpl(
                                pose_values, beta_values)

                            # added 1 to sampled steps so that it is located right after the number.
                            matched_list = [-1] * self.max_len
                            nn0 = [np.argmin(abs(steps1_eff - b))
                                   for b in steps0_eff]
                            nn1 = [np.argmin(abs(steps0_eff - b))
                                   for b in steps1_eff]

                            for ii in range(len0):
                                if ii == nn1[nn0[ii]]:
                                    matched_list[ii] = nn0[ii]
                        else:
                            pose_values = self.data[data_name]['pose']
                            beta_values = self.data[data_name]['beta']

                if NO_TIME_AUG:
                    if self.smpl or self.mano:
                        pose_values = self.data[data_name]['pose']
                        beta_values = self.data[data_name]['beta']

                # geometric augmentation.
                ma_window = 10
                IID = False
                if 'noise_angle' in strategy:
                    # 25 joints, x,y,z, 75 gaussiain
                    vposer_prob0 = np.random.uniform(low=-0, high=1)
                    if vposer_prob0 < 0.3:
                        mu, sigma = 0, 10.0

                        #print("pose_values", pose_values.shape)
                        T, D = pose_values.shape
                        # Augment angles in the Euler space.

                        euler_angle = R.from_rotvec(np.reshape(
                            pose_values, (-1, 3))).as_euler('zyx', degrees=True)
                        M, K = euler_angle.shape
                        euler_angle = np.reshape(euler_angle, (T, -1))

                        T, P = euler_angle.shape

                        cov = (
                            T - np.abs(np.arange(T)[:, np.newaxis] - np.arange(T)[np.newaxis, :])/2) / T
                        if IID:
                            noise_angle = np.random.normal(mu, sigma, (P, T))
                        else:
                            noise_angle = np.random.multivariate_normal(
                                [mu]*T, cov*sigma, P)
                            for ii in range(P):
                                noise_angle[ii] = moving_average(
                                    noise_angle[ii], ma_window)
                        euler_angle = np.reshape(euler_angle, (M, K))  # .T

                        aug_rotvec = np.reshape(R.from_euler(
                            'zyx', euler_angle, degrees=True).as_rotvec(), (len1, -1))

                        pose_values = aug_rotvec
                        if self.smpl:
                            keypoints1[:len1, :, :] = self.get_keypoints_from_smpl(
                                aug_rotvec, beta_values)
                        elif self.mano:
                            keypoints1[:len1, :, :] = self.get_keypoints_from_mano(
                                aug_rotvec, beta_values)

                if "noise_vposer" in strategy:

                    vposer_prob0 = np.random.uniform(low=-0, high=1)
                    if vposer_prob0 < 0.1:

                        mu, sigma = 0, 0.1  # 0.1
                        DIM_VPOSER = 32

                        body_pose = torch.from_numpy(pose_values[:, 3:66]).type(
                            torch.float).to(device=self.device)

                        cov = (len1 - np.abs(np.arange(len1)
                               [:, np.newaxis] - np.arange(len1)[np.newaxis, :])/2) / len1
                        if IID:
                            #noise_angle = np.random.normal(mu, sigma, (P, T))
                            noise_vposer = np.random.normal(
                                mu, sigma, DIM_VPOSER)
                        else:
                            noise_vposer = np.random.multivariate_normal(
                                [mu]*len1, cov*sigma, DIM_VPOSER)
                            for ii in range(DIM_VPOSER):
                                noise_vposer[ii] = moving_average(
                                    noise_vposer[ii], ma_window)

                        noise_vposer = noise_vposer.T

                        body_poZ = self.vp.encode(body_pose).mean
                        body_poZ = body_poZ.T
                        vposer_window = 3
                        vposer_edge_interval = vposer_window//2 + 1
                        temp_start = copy.deepcopy(
                            body_poZ[:, :vposer_edge_interval])
                        temp_end = copy.deepcopy(
                            body_poZ[:, -vposer_edge_interval:])
                        # print("temp_end",temp_end.shape)
                        body_poZ = body_poZ.detach().cpu().numpy()
                        if not IID:
                            for ii in range(DIM_VPOSER):
                                body_poZ[ii] = moving_average(
                                    body_poZ[ii], vposer_window)
                        body_poZ = torch.Tensor(body_poZ).type(
                            torch.float).to(device=self.device)
                        body_poZ[:, :vposer_edge_interval] = temp_start
                        body_poZ[:, -vposer_edge_interval:] = temp_end
                        body_poZ = body_poZ.T

                        # add noise to enconded body poses.
                        body_poZ = body_poZ + \
                            torch.Tensor(noise_vposer).type(
                                torch.float).to(device=self.device)
                        body_pose_rec = self.vp.decode(
                            body_poZ)['pose_body'].contiguous().view(-1, 63)
                        body_pose_rec = body_pose_rec.detach().cpu().numpy()

                        body_pose_rec = np.concatenate(
                            (pose_values[:, :3], body_pose_rec, pose_values[:, 66:]), axis=1)
                        pose_values = body_pose_rec
                        # print("body_pose_rec",body_pose_rec.shape)
                        keypoints1[:len1, :, :] = self.get_keypoints_from_smpl(
                            body_pose_rec, beta_values)


                if ('flip' in strategy):

                    flip_prob1 = np.random.uniform(low=-0, high=1)
                    if flip_prob1 < 0.3:
                        keypoints1 = keypoints1 * [-1, 1, 1]
                        keypoints1 = keypoints1.astype(np.float32)
                        if self.smpl:
                            keypoints1[:len1] = pre_normalization_mat(keypoints1[:len1],
                                                                      zaxis=[1, 8], xaxis=[1, 5])
                        elif self.mano:
                            keypoints1[:len1] = pre_normalization_mat(keypoints1[:len1],
                                                                      zaxis=[0, 1], xaxis=[0, 9])
                        else:
                            keypoints1[:len1] = pre_normalization_mat(keypoints1[:len1],
                                                                      zaxis=[5, 11], xaxis=[5, 6], ERR_BORN=True)

                # and (vposer_prob0 > 0.3):
                if ('noise_translation' in strategy):
                    # make noise translation and vposer not working together.
                    vposer_prob0 = np.random.uniform(low=-0, high=1)
                    if vposer_prob0 < 0.3:
                        # 25 joints, x,y,z, 75 gaussiain
                        T, K, D = keypoints1.shape
                        mu, sigma = 0, 0.1  # 0.1
                        noise_trans = np.random.normal(mu, sigma, (T, K, D))
                        keypoints1 = keypoints1 + noise_trans

                VIS = False
                if VIS:

                    vis_folder = 'vis/motion/key0'
                    s = copy.deepcopy(keypoints0)
                    s = s * [1, -1, -1]
                    s1 = copy.deepcopy(keypoints1)
                    s1 = s1 * [1, -1, -1]
                    fig = plt.figure()
                    connectivity = get_openpose_connectivity()
                    ax = plt.axes(projection='3d')

                    for i_s in range(len0):
                        ax.set_xlim3d(-2, 2)
                        ax.set_ylim3d(-2, 2)
                        ax.set_zlim3d(-2, 2)

                        for limb in connectivity:
                            ax.plot3D(s[i_s, limb, 0],
                                      s[i_s, limb, 1], s[i_s, limb, 2], 'red')
                        ax.scatter3D(s[i_s, :, 0], s[i_s, :, 1],
                                     s[i_s, :, 2], cmap='Greens', s=2)

                        for limb in connectivity:
                            ax.plot3D(s1[i_s, limb, 0],
                                      s1[i_s, limb, 1], s1[i_s, limb, 2], 'blue')
                        ax.scatter3D(s1[i_s, :, 0], s1[i_s, :, 1],
                                     s1[i_s, :, 2], cmap='Greens', s=2)

                        plt.show(block=False)
                        plt.savefig(vis_folder + "/file%04d.png" % i_s)
                        plt.cla()

                    os.chdir("vis/motion/key0")
                    subprocess.call([
                        'ffmpeg', '-framerate', '8', '-i', 'file%04d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                        'video_name.mp4'
                    ])
                    sys.exit(0)

                data = {
                    # 'valid_mask' : mask,
                    'len0': len0,
                    'len1': len1,
                    'steps0': steps0,
                    'steps1': steps1,
                    'mode': self.mode_str,
                    'keypoints0': keypoints0,   # (1, h, w)
                    'label0': label0,   # (h, w)
                    'keypoints1': keypoints1,
                    'label1': label1,
                    'matching': torch.IntTensor(matched_list),
                    'dataset_name': 'PennAction',
                    'pair_id': idx,
                    'pair_names': data_name
                }
                return data

            
        else:
    
            label0 = self.data[data_name]['labels']
            len0 = len(label0)
            if self.sampling_strategy == 'offset_uniform':
                steps0 = list(range(len0))
                steps0 = torch.reshape(torch.stack(
                    list(map(self.get_steps, steps0))), [-1])
                steps0 = torch.maximum(steps0, torch.tensor(0))
                steps0 = torch.minimum(steps0, torch.tensor(len0-1))
            keypoints0 = self.data[data_name]['pose'][steps0]
            label0 = self.data[data_name]['labels'][steps0]
            data = {
                'len0': len0,
                'steps0': steps0,
                'mode': self.mode_str,
                'keypoints0': keypoints0,   # (1, h, w)
                'label0': label0,   # (h, w)
                'dataset_name': 'PennAction',
                'pair_id': idx,
                'pair_names': data_name
            }

        return data
