import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import pandas as pd
import os


def vis_conf_matrix(conf_matrix, save_path):

    conf_matrix = np.around(conf_matrix, decimals=5)
    sn.set(font_scale=0.05)
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in range(np.shape(conf_matrix)[0])],
                         columns=[i for i in range(np.shape(conf_matrix)[1])])
    df_cm = df_cm[::-1]
    svm = sn.heatmap(df_cm, annot=False, cmap="OrRd")
    figure = svm.get_figure()
    figure.savefig(save_path, dpi=400)
    figure.clf()
