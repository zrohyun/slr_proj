# check skeleton data
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import pathlib


def annotate(axis, text, x, y):
    text_annotation = Annotation(text, xy=(x, y), xycoords="data")
    axis.add_artist(text_annotation)


def get_sample_data():
    with open(pathlib.Path(__file__).parent.resolve() / "sample_data.txt", "r") as f:
        sample_data = dict(eval(f.read()))
        sample_data = sample_data["people"]
        ks = [
            "face_keypoints_2d",
            "pose_keypoints_2d",
            "hand_left_keypoints_2d",
            "hand_right_keypoints_2d",
        ]
        return np.concatenate(
            tuple(np.array(sample_data[k]).reshape((-1, 3)) for k in ks)
        )


def plot_scatt(x, y, v_range, annot=True, savefig=True, fig_name="fig.png"):
    figure, ax = plt.subplots(figsize=(10, 10))
    x_ = x[v_range[0] : v_range[-1]]
    y_ = y[v_range[0] : v_range[-1]]
    ax.scatter(x_, y_)
    if annot:
        for i in v_range:
            annotate(ax, str(i), x[i], y[i])
    ax.invert_yaxis()
    if savefig:
        plt.savefig(fig_name)
    plt.show()


if __name__ == "__main__":
    skel = get_sample_data()

    plot_scatt(skel[:, 0], skel[:, 1], range(0, 70), fig_name="face.png")
    plot_scatt(skel[:, 0], skel[:, 1], range(70, 95), fig_name="pose.png")
    plot_scatt(skel[:, 0], skel[:, 1], range(95, 116), fig_name="l_hands.png")
    plot_scatt(skel[:, 0], skel[:, 1], range(116, 137), fig_name="r_hands.png")
