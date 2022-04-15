import copy
from tqdm import tqdm
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.extend(['../'])
'''IKEA ASM
        "nose", # 0
        "left eye", # 1
        "right eye", # 2
        "left ear", # 3
        "right ear", # 4
        "left shoulder", # 5 - center
        "right shoulder", # 6
        "left elbow", # 7
        "right elbow", # 8
        "left wrist", # 9
        "right wrist", # 10
        "left hip", # 11 
        "right hip", # 12
        "left knee", # 13
        "right knee", # 14
        "left ankle", # 15
        "right ankle", # 16
'''


def get_openpose_connectivity():
    return [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 5],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 24],
        [11, 22],
        [22, 23],
        [8, 12],
        [12, 13],
        [13, 14],
        [14, 21],
        [14, 19],
        [19, 20],
        [0, 15],
        [15, 17],
        [0, 16],
        [16, 18]
    ]


def get_ikea_connectivity():
    return [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [0, 5],
        [0, 6],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [5, 11],
        [6, 12],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16]
    ]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                 [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [
                 0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)


def pre_normalization(data, zaxis=[5, 11], xaxis=[5, 6], NORM_BONE=True):
    """1. Normalize 1st frame.
       2. Normalize every frame.
       3. Normalize every frame but leave the first location to give more information.
    """
    VIS = False

    # print("data.shape", np.shape(data))  # (993, 17, 3)
    data = np.array(data)[:, :, :3]
    N, K, C = data.shape  # (993, 17, 3)
    s = data
    # print('sub the center joint #5 (left shoulder)')
    for i_s, skeleton in enumerate(s):

        if np.sum(skeleton[zaxis[0]]) == 0:
            continue
        # Instaed of setting the center as #5, we can also cal mean vals.
        main_body_center = skeleton[zaxis[0]:zaxis[0]+1, :].copy()
        s[i_s] = s[i_s] - main_body_center

    # initialization
    joint_bottom_prev = np.array([0, 0, 0])
    joint_top_prev = np.array([0, 0, 0])
    #print('parallel the edge between left shoulder(5) and right shoulder(6) to the x axis')

    for i_s, skeleton in enumerate(s):
        if np.sum(skeleton) == 0:
            continue
        if np.sum(skeleton[xaxis[1]]) == 0:
            #print("x axis is not exist", i_s)
            # if joint in xaxis[1] is zero, then we will use from the previous frame's
            # one because it will be not much different.
            joint_bottom = joint_bottom_prev
            joint_top = joint_top_prev
        else:
            joint_bottom = skeleton[xaxis[0]]
            joint_top = skeleton[xaxis[1]]
            joint_bottom_prev = joint_bottom
            joint_top_prev = joint_top
        joint_rshoulder = skeleton[xaxis[0]]
        joint_lshoulder = skeleton[xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)

        for i_j, joint in enumerate(skeleton):
            s[i_s, i_j] = np.dot(matrix_x, joint)

    #print('parallel the edge between left shoulder(5) and left heap(11) to the z axis')

    for i_s, skeleton in enumerate(s):
        if np.sum(skeleton) == 0:
            continue
        if np.sum(skeleton[zaxis[1]]) == 0:
            #print("z axis is not exist", i_s)
            # if joint in zaxis[1] is zero, then we will use from the previous frame's
            # one because it will be not much different.
            joint_bottom = joint_bottom_prev
            joint_top = joint_top_prev
        else:
            joint_bottom = skeleton[zaxis[0]]
            joint_top = skeleton[zaxis[1]]
            joint_bottom_prev = joint_bottom
            joint_top_prev = joint_top
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        # print("joint_top", joint_top)
        # print("joint_bottom", joint_bottom)
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_j, joint in enumerate(skeleton):
            s[i_s, i_j] = np.dot(matrix_z, joint)

    # Normalize bones
    if NORM_BONE:
        for i_s, skeleton in enumerate(s):
            bone_length = np.linalg.norm(
                skeleton[zaxis[0]] - skeleton[zaxis[1]])
            if (np.sum(skeleton) == 0) or (bone_length == 0):
                continue

            #print("bone_length", bone_length)
            #print("s[i_s] before", s[i_s])
            s[i_s] = s[i_s]/bone_length
            #print("s[i_s] after", s[i_s])
            if VIS:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.set_xlim3d(-1, 1)
                ax.set_ylim3d(-1, 1)
                ax.set_zlim3d(-1, 1)
                connectivity = get_openpose_connectivity()
                for limb in connectivity:
                    ax.plot3D(s[i_s, limb, 0],
                              s[i_s, limb, 1], s[i_s, limb, 2])
                ax.scatter3D(s[i_s, :, 0], s[i_s, :, 1],
                             s[i_s, :, 2], cmap='Greens')
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()

    # print("s", np.shape(s))
    return s


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
