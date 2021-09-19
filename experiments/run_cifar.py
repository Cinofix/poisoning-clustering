from run_multiway_clustering import test_robustness
import torch
import numpy as np
from clustorch.kmeans import KMeans
from clustorch.spectral import SpectralClustering
from clustorch.hierarchical import Hierarchical
from experiments.device import opts

DEVICE = opts.device
PATH = opts.path
PATH += "/pr2021/"


def main():

    device = DEVICE

    x = torch.load("./data/cifar10/cifar_features_classes016.pt")
    y = torch.load("./data/cifar10/cifar_labels_classes016.pt")

    X = x.unsqueeze(2).to(device)
    Y = y.to(device)

    dt_range = np.linspace(start=0.01, stop=1.5, num=20)
    s_range = np.linspace(start=0.01, num=20, stop=0.6)

    k = len(Y.unique())

    models = [
        KMeans(n_clusters=k, max_tol=1e-05, max_iter=500),
        SpectralClustering(
            n_clusters=k, lmode="rw", similarity="gaussian_zp", assign_labels="kmeans",
        ),
        Hierarchical(n_clusters=k),
    ]

    test_robustness(
        X,
        Y,
        y_target=6,
        models=models,
        dt_range=dt_range,
        box=(0, 6),
        s_range=s_range,
        lb=1 / 255,
        path=PATH + "cifar/",
    )


if __name__ == "__main__":
    main()
