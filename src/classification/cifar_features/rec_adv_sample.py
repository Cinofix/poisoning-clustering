import sys

root = "../../../"
sys.path.extend([root])

from experiments.utilities import *

import torch
import numpy as np
from clustorch.kmeans import KMeans
from src.classification.cifar_features.pytorchcifar.models import ResNet18
from src.classification.cifar_features.extract_features import convert_keys
from experiments.device import opts
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal

DEVICE = opts.device
PATH = opts.path
PATH += "/pr2021/"


def shuffle_indexes(idxs):
    return idxs[torch.randperm(len(idxs))]


def concat(x, y):
    if x is None:
        return y
    return torch.cat([x, y])


def transform(net, loader, device="cpu"):
    net.eval()
    net.to(device)

    ds_x = None
    ds_y = None
    for data, labels in loader:
        mask = (labels == 0) | (labels == 1) | (labels == 6)
        x, y = data[mask], labels[mask]
        x = x.to(device)
        y = y.to(device)
        fx = net.get_features(x)
        ds_x = concat(ds_x, fx.cpu())
        ds_y = concat(ds_y, y.cpu())
        if ds_x.shape[0] > 800:
            return ds_x, ds_y

    return ds_x, ds_y


device = DEVICE
x = torch.load(root + "data/cifar10/cifar_features_classes016.pt")
y = torch.load(root + "data/cifar10/cifar_labels_classes016.pt")

X = x.unsqueeze(2).to(device)
Y = y.to(device)

k = len(Y.unique())

model = KMeans(n_clusters=k, max_tol=1e-05, max_iter=500)
model = ClusteringWrapper3Dto2D(model)

# Reconstruct to input space
net = ResNet18()
checkpoint = torch.load("./pytorchcifar/checkpoint/ckpt_resnet18.pth")
weights = convert_keys(checkpoint["net"])
net.load_state_dict(weights)

import torchvision
import torchvision.transforms as transforms

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

ds = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)


# fx_ds, fy_ds = transform(net, loader, device="cpu")
cx, cy = next(loader.__iter__())

from src.classification.cifar_features.rec_utilities import reconstruct

net.cuda()
"""

for i in range(10):
    xi = cx[cy == i][0].clone().to("cuda:0")
    fx = fx_ds[0].to("cuda:0")#net.get_features(cx[cy == i][0].cuda().unsqueeze(0))  # X[ts_idx]
    x_poison, x_opt_up, rec_error = reconstruct(
        net.to("cuda:0"), x_i=xi, fx_p=fx, lr=0.01, epochs=10, device="cuda:0"
    )
"""
import matplotlib.pyplot as plt

_, axs = plt.subplots(1, 3, figsize=(16, 6))
axs = axs.flatten()

for i, target in zip(range(3), [[0, 1], [1, 0], [6, 0]]):
    T = ConstrainedAdvPoisoningGlobal(
        delta=0.1,
        s=1,
        clst_model=model,
        lb=1 / 255,
        G=110,
        domain_cons=(X.min(), X.max()),
        objective="AMI",
        mode="guided",
        link="centroids",
        mutation_rate=0.05,
        crossover_rate=0.85,
        zero_rate=0.001,
    )

    Xadv, leps, ts_idx, direction = T.forward(X, Y, target)
    xi = cx[cy == target[0]][0].clone().to("cuda:0")
    fx = Xadv[ts_idx].flatten()
    x_poison, x_opt_up, rec_error = reconstruct(
        net, x_i=xi, fx_p=fx, lr=0.01, epochs=500, device="cuda:0"
    )
    axs[i].imshow(x_poison.cpu().numpy().transpose(1, 2, 0))

plt.savefig("poison_samples_01.pdf", bbox_inches="tight", dpi=1000)
plt.show()
