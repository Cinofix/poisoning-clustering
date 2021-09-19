from run_experiments_export import test_robustness
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import numpy as np
from clustorch.kmeans import KMeans
from clustorch.spectral import SpectralClustering
from clustorch.hierarchical import Hierarchical
from experiments.run_fashion import filter_by_label
from experiments.device import opts

DEVICE = opts.device
PATH = opts.path

PATH += "/pr2021/"


def main():
    root = "./data/"
    trans = transforms.Compose([transforms.ToTensor(),])
    train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)

    torch.manual_seed(4)
    n_samples = 800

    dt_range = np.linspace(start=0.1, num=19, stop=1)
    s_range = np.linspace(start=0.1, num=21, stop=0.6)

    X, Y = filter_by_label(
        x=train_set.data,
        y=train_set.targets,
        labels=[6, 9],
        n_samples=n_samples,
        device=DEVICE,
    )
    k = len(Y.unique())
    models = [
        KMeans(n_clusters=k),
        SpectralClustering(
            n_clusters=k, lmode="rw", similarity="gaussian_zp", assign_labels="kmeans",
        ),
        Hierarchical(n_clusters=k),
    ]
    test_robustness(
        X,
        Y,
        models=models,
        dt_range=dt_range,
        s_range=s_range,
        lb=1 / 255,
        path="./exportfashionMNIST/",
    )


if __name__ == "__main__":
    main()
