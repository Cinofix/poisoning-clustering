import numpy as np
import matplotlib.pyplot as plt
import torch


class SKlearnClusteringWrapper:
    def __init__(self, model):
        self.model = model

    def fit_predict(self, X):
        return torch.from_numpy(self.model.fit_predict(X.squeeze(2)))

    def to_string(self):
        return self.model.to_string()


def comparison(X, yhat, model, T, N=20):
    miss_clustered = np.zeros(N)
    l2 = np.zeros(N)
    linf = np.zeros(N)
    l0 = np.zeros(N)
    for i in range(N):
        Xadv_m, leps, ts_idx, d = T.forward(X, yhat, from_to=[1, 0])
        yadv_m = model.fit_predict(Xadv_m)
        miss_clustered[i] = min(
            (yhat != yadv_m).sum(), (yhat != (1 - yadv_m)).sum()
        ).item()
        l2[i] = (Xadv_m - X).norm(2).item()
        l0[i] = (Xadv_m - X).norm(0).item()
        linf[i] = (Xadv_m - X).norm(float("inf")).item()
    return miss_clustered, l0, l2, linf


def show_fig(X, Xadv_s, Xadv_m, idx_m, idx_s, shape=(8, 8), out_file="digit_14.png"):
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, axs = plt.subplots(1, 3)
    plt.rcParams["figure.figsize"] = (12, 12)
    im1 = axs[0].imshow(X[idx_m].view(shape), cmap="gray_r")
    im1.axes.get_xaxis().set_visible(False)
    im1.axes.get_yaxis().set_visible(False)
    axs[0].set_title("Target Sample", fontsize=16)
    cl = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cl.ax.tick_params(labelsize=11)

    im3 = axs[1].imshow(Xadv_s[idx_s].view(shape), cmap="gray_r")
    im3.axes.get_xaxis().set_visible(False)
    im3.axes.get_yaxis().set_visible(False)
    axs[1].set_title("Spill-over [26]", fontsize=16)
    cl = fig.colorbar(im3, ax=axs[1], fraction=0.046, pad=0.04)
    cl.ax.tick_params(labelsize=11)

    im4 = axs[2].imshow(Xadv_m[idx_m].view(shape), cmap="gray_r")
    im4.axes.get_xaxis().set_visible(False)
    im4.axes.get_yaxis().set_visible(False)
    axs[2].set_title("Ours", fontsize=16)
    cl = fig.colorbar(im4, ax=axs[2], fraction=0.046, pad=0.04)
    cl.ax.tick_params(labelsize=11)
    plt.savefig(out_file)
