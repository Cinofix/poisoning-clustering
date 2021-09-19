from run_experiments_transfer import test_robustness
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import numpy as np
from clustorch.kmeans import KMeans
from clustorch.spectral import SpectralClustering
from clustorch.hierarchical import Hierarchical
from experiments.device import opts
from experiments.run_fashion import shuffle_indexes, filter_by_label

DEVICE = opts.device
PATH = opts.path

PATH += "/pr2021/"


def split_train_val(X_train_val, Y_train_val, n_samples):
    n, m, _ = X_train_val.shape
    n_in_class = n_samples // 2
    idxs = shuffle_indexes(torch.arange(n))

    X_train, Y_train = X_train_val[:n_in_class], Y_train_val[:n_in_class]
    X_train = torch.cat([X_train, X_train_val[n_samples : n_samples + n_in_class]])
    Y_train = torch.cat([Y_train, Y_train_val[n_samples : n_samples + n_in_class]])

    X_val, Y_val = (
        X_train_val[n_in_class:n_samples],
        Y_train_val[n_in_class:n_samples],
    )
    X_val = torch.cat([X_val, X_train_val[-n_in_class:]])
    Y_val = torch.cat([Y_val, Y_train_val[-n_in_class:]])

    return X_train, Y_train, X_val, Y_val


def main():
    root = "./data/"
    trans = transforms.Compose([transforms.ToTensor(),])
    train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)

    torch.manual_seed(4)
    n_samples = 1600

    dt_range = np.linspace(start=0.05, num=20, stop=1)
    s_range = np.linspace(start=0.01, num=20, stop=0.6)

    X_train_val, Y_train_val = filter_by_label(
        x=train_set.data,
        y=train_set.targets,
        labels=[6, 9],
        n_samples=n_samples,
        device=DEVICE,
    )

    X, Y, X_transf, Y_transf = split_train_val(X_train_val, Y_train_val, n_samples)
    k = len(Y.unique())
    print(Y.sum(), Y_transf.sum())
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
        X_transf,
        Y_transf,
        models=models,
        dt_range=dt_range,
        s_range=s_range,
        lb=1 / 255,
        mutation_rate=0.1,
        path=PATH + "/TransferfashionMNIST/",
    )


if __name__ == "__main__":
    main()
