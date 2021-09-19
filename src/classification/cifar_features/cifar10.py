import sys

sys.path.extend(["./"])

from clustorch.kmeans import KMeans
from clustorch.hierarchical import Hierarchical
from clustorch.spectral import SpectralClustering
from experiments.utilities import *


def filter_data(dt_x, dt_y, first, second, third=None, n=250):
    if third is not None:
        mask = (dt_y == first) | (dt_y == second) | (dt_y == third)
    else:
        mask = (dt_y == first) | (dt_y == second)

    # mask = (train_set.train_labels == 1) | (train_set.train_labels == 6)
    idxs = torch.nonzero(mask, as_tuple=False)
    perm = idxs[torch.randperm(len(idxs))][:n]
    X = dt_x[perm].flatten(1)
    Y = dt_y[perm].flatten(0)
    return X, Y


X = torch.load(
    "./src/classification/cifar_features/cifar_features_resnet18.pt"
).detach()
Y = torch.load("./src/classification/cifar_features/cifar_labels.pt").detach()

# best_tp = (6, 1, 7)
# best_tp = (4, 6, 9)
best_tp = (0, 1, 6)
i, j, k = best_tp
set_seed(4)
x, y = filter_data(X, Y, i, j, k, n=1600)
k = len(y.unique())
models = [
    KMeans(n_clusters=k, max_tol=1e-05, max_iter=500),
    SpectralClustering(
        n_clusters=k, lmode="rw", similarity="gaussian_zp", assign_labels="kmeans"
    ),
    Hierarchical(n_clusters=k),
]

for m in models:
    cls = m.fit_predict(x)
    print(adjusted_mutual_info_score(cls, y))
torch.save(x, "./data/cifar10/cifar_features_classes016.pt")
torch.save(y, "./data/cifar10/cifar_labels_classes016.pt")
