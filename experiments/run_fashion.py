from run_experiments import test_robustness
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from sklearn import preprocessing
import numpy as np
from clustorch.kmeans import KMeans
from clustorch.spectral import SpectralClustering
from clustorch.hierarchical import Hierarchical
from experiments.device import opts
from experiments.utilities import shuffle_indexes

DEVICE = opts.device
PATH = opts.path

PATH += "/pr2021/"


def filter_by_label(x, y, labels, n_samples, encode=True, device="cpu"):
    le = preprocessing.LabelEncoder()
    to_pick = torch.tensor([], dtype=torch.long)
    for l in labels:
        mask = y == l
        valid = torch.nonzero(mask, as_tuple=False)
        to_pick_l = shuffle_indexes(valid)[:n_samples]
        to_pick = torch.cat((to_pick, to_pick_l), dim=0)
    X = x[to_pick].view(-1, 28 * 28).float()
    X = X.unsqueeze(2).to(device)
    X /= 255.0
    Y = y[to_pick].view(-1)
    if encode:
        Y = torch.from_numpy(le.fit_transform(Y))
    Y = Y.to(device)
    return X, Y


def main():
    root = "./data/"
    trans = transforms.Compose([transforms.ToTensor(),])
    train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)

    torch.manual_seed(4)
    n_samples = 800

    dt_range = np.linspace(start=0.05, num=20, stop=1)
    s_range = np.linspace(start=0.01, num=20, stop=0.6)

    X, Y = filter_by_label(
        x=train_set.data,
        y=train_set.targets,
        labels=[6, 9],  # [9,3]
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
        mutation_rate=0.1,
        path=PATH + "/fashionMNIST/",
    )


if __name__ == "__main__":
    main()
