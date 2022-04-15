import json
import scipy.io
import os
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_penn_connectivity():
    return [
        [0, 1],
        [1, 4],
        [4, 7],
        [7, 10],
        [0, 2],
        [2, 5],
        [5, 8],
        [8, 11],
        [0, 3],
        [3, 6],
        [6, 9],
        [9, 13],
        [13, 16],
        [16, 18],
        [18, 20],
        [20, 22],
        [9, 12],
        [12, 15],
        [9, 14],
        [14, 17],
        [17, 19],
        [19, 21],
        [21, 23]
    ]


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


if __name__ == "__main__":
    VIS = True
    dataset_path = ''
    label_path = os.path.join(dataset_path, 'labels')
    bbox_path = os.path.join(dataset_path, 'bbox')
    mocap_path = os.path.join(dataset_path, 'mocap')

    # frame = 1
    pose_path = os.path.join(mocap_path, "0001/mocap")
    for frame in range(1, 100):
        with open(os.path.join(pose_path, '{0:06d}_prediction_result.pkl'.format(frame)), 'rb') as f:
            data = pickle.load(f)

        # body_pose = np.reshape(
        #    (data['pred_output_list'][0]['pred_body_pose'][0]), (24, 3))  # (24, 3)
        # 25,3
        body_pose = data['pred_output_list'][0]['pred_body_joints_img'][:25]
        # print("body_pose", body_pose)

        if VIS:
            s = body_pose
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlim3d(-200, 200)
            ax.set_ylim3d(-200, 200)
            ax.set_zlim3d(-200, 200)
            connectivity = get_openpose_connectivity()
            for limb in connectivity:
                ax.plot3D(s[limb, 0], s[limb, 1], s[limb, 2])
            ax.scatter3D(s[:, 0], s[:, 1],
                         s[:, 2], cmap='Greens')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
    # visualize
