import copy
from tqdm import tqdm
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.spatial.transform import Rotation as R

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

    
def get_h2o_connectivity():
    offset = 21
    return [
        [1,2],
        [2,3],
        [3,4],
        [5,6],
        [6,7],
        [7,8],
        [9,10],
        [10,11],
        [11,12],
        
        [13,14],
        [14,15],
        [15,16],
        [17,18],
        [18,19],
        [19,20],

        [0,1],
        [0,5],
        [0,9],
        [0,13],
        [0,17],

        [1+offset,2+offset],
        [2+offset,3+offset],
        [3+offset,4+offset],
        [5+offset,6+offset],
        [6+offset,7+offset],
        [7+offset,8+offset],
        [9+offset,10+offset],
        [10+offset,11+offset],
        [11+offset,12+offset],
        
        [13+offset,14+offset],
        [14+offset,15+offset],
        [15+offset,16+offset],
        [17+offset,18+offset],
        [18+offset,19+offset],
        [19+offset,20+offset],

        [0+offset,1+offset],
        [0+offset,5+offset],
        [0+offset,9+offset],
        [0+offset,13+offset],
        [0+offset,17+offset],


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
    # print("np.linalg.norm(vector)", np.linalg.norm(vector))
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
    # print("v1", v1)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # print("v1_u", v1_u)
    # print("v2_u", v2_u)
    # print("np.dot(v1_u, v2_u)", np.dot(v1_u, v2_u))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def unit_vector_mat(vector):
    """ Returns the unit vector of the vector.  """
    # print("matamt")
    # print("np.linalg.norm(vector)", np.linalg.norm(vector, axis=1))
    # print("np.linalg.norm(vector, axis=1)[0]",
    #      np.linalg.norm(vector, axis=1))
    return (vector.T/np.linalg.norm(vector, axis=1).T).T


def angle_between_mat(v1, v2):
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
    # print("v1", v1)
    v1_u = unit_vector_mat(v1)
    v2_u = unit_vector_mat(v2)

    # print("v1_u", v1_u)
    # print("v2_u", v2_u)
    # print("np.multiply(v1_u, v2_u)", np.sum(np.multiply(v1_u, v2_u), axis=1))
    return np.arccos(np.clip(np.sum(np.multiply(v1_u, v2_u), axis=1), -1.0, 1.0))


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

# ikea 5,11 / 5,6   openpose 1,8/1,5
def pre_normalization_mat(data, zaxis=[5, 11], xaxis=[5, 6], NORM_BONE=True,ERR_BORN=False):
    """1. Normalize 1st frame.
       2. Normalize every frame.
       3. Normalize every frame but leave the first location to give more information.
    """
    VIS = False

    # print("data.shape", np.shape(data))  # (993, 17, 3)
    data = np.array(data)[:, :, :3]
    N, K, C = data.shape  # (993, 17, 3)

    # print('sub the center joint #5 (left shoulder)')
    # for i_s, skeleton in enumerate(s):

    #    if np.sum(skeleton[zaxis[0]]) == 0:
    #        continue
    # Instaed of setting the center as #5, we can also cal mean vals.

    main_body_center = data[:, zaxis[0]:zaxis[0]+1, :]
    data = data - main_body_center

    # print('parallel the edge between left shoulder(5) and right shoulder(6) to the x axis')
    joint_bottom = data[:, xaxis[0]]
    joint_top = data[:, xaxis[1]]
    joint_rshoulder = data[:, xaxis[0]]
    joint_lshoulder = data[:, xaxis[1]]
    axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    input_axis = np.tile([1, 0, 0], (N, 1))
    angle = angle_between_mat(joint_rshoulder - joint_lshoulder, input_axis)
    #print("angle", angle)
    # r = R.from_rotvec((input_axis.T*angle.T).T)

    for ii in range(N):
        matrix_x = rotation_matrix(axis[ii], angle[ii])
        data[ii] = np.dot(matrix_x, data[ii].T).T

    # print('parallel the edge between left shoulder(5) and left heap(11) to the z axis')
    joint_bottom = data[:, zaxis[0]]
    joint_top = data[:, zaxis[1]]
    # oint_rshoulder = data[:, xaxis[0]]
    # joint_lshoulder = data[:, xaxis[1]]
    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    input_axis = np.tile([0, 0, 1], (N, 1))
    angle = angle_between_mat(joint_top - joint_bottom, input_axis)
    # print("angle", angle)
    # r = R.from_rotvec((input_axis.T*angle.T).T)

    for ii in range(N):
        matrix_z = rotation_matrix(axis[ii], angle[ii])
        data[ii] = np.dot(matrix_z, data[ii].T).T

    # Normalize bones
    if NORM_BONE:
        if ERR_BORN:
            s = copy.deepcopy(data)
            for i_s, skeleton in enumerate(s):
                bone_length = np.linalg.norm(
                    skeleton[zaxis[0]] - skeleton[zaxis[1]])
                if (np.sum(skeleton) == 0) or (bone_length == 0):
                    continue

            data = s
        else:
            
            bone_length = np.linalg.norm(
                data[:, zaxis[0]] - data[:, zaxis[1]], axis=1)
            data = data/np.tile(np.expand_dims(bone_length,
                                            axis=(1, 2)), (1, K, C))
            #    print("bone_length", bone_length)

        # print("bone_length", bone_length)
        # print("s[i_s] before", s[i_s])
        #    s[i_s] = s[i_s]/bone_length
        # print("s[i_s] after", s[i_s])
        

        # print("np.expand_dims(bone_lengthaxis(1, 2)", np.expand_dims(bone_length,
        #                                                             axis=(1, 2)))

        #print("s", s)
        #print("data", data)

    # if VIS:
    #    for i_s, skeleton in enumerate(s):
    #            fig = plt.figure()
    #            ax = plt.axes(projection='3d')
    #            ax.set_xlim3d(-1, 1)
    #            ax.set_ylim3d(-1, 1)
    #            ax.set_zlim3d(-1, 1)
    #            connectivity = get_openpose_connectivity()
    #            for limb in connectivity:
    #                ax.plot3D(s[i_s, limb, 0],
    #                          s[i_s, limb, 1], s[i_s, limb, 2])
    #            ax.scatter3D(s[i_s, :, 0], s[i_s, :, 1],
    #                         s[i_s, :, 2], cmap='Greens')
    #            plt.show(block=False)
    #            plt.pause(0.5)
    #            plt.close()

    # print("s", np.shape(s))
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
