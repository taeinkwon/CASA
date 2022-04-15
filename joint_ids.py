# Joint Ids and it's connectivity

import numpy as np

def get_joint_names_dict(joint_names):
    return {name: i for i, name in enumerate(joint_names)}

def get_ikea_joint_names():
    return [
        "nose", # 0
        "left eye", # 1
        "right eye", # 2
        "left ear", # 3
        "right ear", # 4
        "left shoulder", # 5
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

def get_body25_joint_names():
    return [
        "nose", # 0
        "neck", # 1
        "right shoulder", # 2
        "right elbow", # 3
        "right wrist", # 4
        "left shoulder", # 5
        "left elbow", # 6
        "left wrist", # 7
        "mid hip", # 8
        "right hip", # 9
        "right knee", # 10
        "right ankle", # 11
        "left hip", # 12
        "left knee", # 13
        "left ankle", # 14
        "right eye", # 15
        "left eye", # 16
        "right ear", # 17
        "left ear", # 18
        "left big toe", # 19
        "left small toe", # 20
        "left heel", # 21
        "right big toe", # 22
        "right small toe", # 23
        "right heel", # 24
        "background", # 25
    ]

def get_body25_connectivity():
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
        [8, 12],
        [12, 13],
        [13, 14],
        [0, 15],
        [0, 16],
        [15, 17],
        [16, 18],
        [2, 9],
        [5, 12],
        [11, 22],
        [11, 23],
        [11, 24],
        [14, 19],
        [14, 20],
        [14, 21],
    ]


def get_body21_joint_names():
    return [
        "nose", # 0
        "neck", # 1
        "right shoulder", # 2
        "right elbow", # 3
        "right wrist", # 4
        "left shoulder", # 5
        "left elbow", # 6
        "left wrist", # 7
        "mid hip", # 8
        "right hip", # 9
        "right knee", # 10
        "right ankle", # 11
        "left hip", # 12
        "left knee", # 13
        "left ankle", # 14
        "right eye", # 15
        "left eye", # 16
        "right ear", # 17
        "left ear", # 18
        "neck (lsp)", # 19
        "top of head (lsp)", # 20
    ]

def get_hmmr_joint_names():
    return [
        "right ankle", # 0
        "right knee", # 1
        "right hip", # 2
        "left hip", # 3
        "left knee", # 4
        "left ankle", # 5
        "right wrist", # 6
        "right elbow", # 7
        "right shoulder", # 8
        "left shoulder", # 9
        "left elbow", # 10
        "left wrist", # 11
        "neck", # 12
        "top of head", # 13
        "nose", # 14
        "left eye", # 15
        "right eye", # 16
        "left ear", # 17
        "right ear", # 18
        "left big toe", # 19
        "right big toe", # 20
        "left small toe", # 21
        "right small toe", # 22
        "left heel", # 23
        "right heel", # 24
    ]
