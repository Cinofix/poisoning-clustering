import sys

sys.path.extend(["./"])

import numpy as np
import torch
from clustorch.kmeans import KMeans
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal
from experiments.utilities import ClusteringWrapper3Dto2D, set_seed


X = np.load("comparison/SEEDS/kme_X_org.npy")
Xadv_s = np.load("comparison/SEEDS/kme_X_adv.npy")

X = torch.from_numpy(X).unsqueeze(2)
Xadv_s = torch.from_numpy(Xadv_s).unsqueeze(2)
eps_s = Xadv_s - X
set_seed(2)

h = KMeans(n_clusters=2)
model = ClusteringWrapper3Dto2D(h)
yhat = model.fit_predict(X)
yadv_s = model.fit_predict(Xadv_s)

print("Suspicion Miss:", min((yhat != yadv_s).sum(), (1 - yhat != yadv_s).sum()))

idx_s = torch.nonzero(Xadv_s - X, as_tuple=False)[0, 0]
print("target sample", idx_s)

import pandas as pd


def comparison(T, N=20):
    miss_clustered = np.zeros(N)
    l2 = np.zeros(N)
    linf = np.zeros(N)
    l0 = np.zeros(N)
    for i in range(N):
        Xadv_m, leps, ts_idx, d = T.forward(X, yhat, from_to=[0, 1])

        yadv_m = model.fit_predict(Xadv_m)
        miss_clustered[i] = min(
            (yhat != yadv_m).sum(), (yhat != (1 - yadv_m)).sum()
        ).item()
        l2[i] = (Xadv_m - X).norm(2).item()
        l0[i] = (Xadv_m - X).norm(0).item()
        linf[i] = (Xadv_m - X).norm(float("inf")).item()
    return miss_clustered, l0, l2, linf


set_seed(4)
constrained = ConstrainedAdvPoisoningGlobal(
    delta=(Xadv_s - X).norm(float("inf")),
    s=1,
    clst_model=model,
    lb=1,
    G=20,
    mutation_rate=0.01,
    crossover_rate=0.85,
    zero_rate=0.1,
    domain_cons=[0, X.max() + 5],
    objective="AMI",
    mode="guided",
    link="centroids",
)
miss, l0, l2, linf = comparison(constrained, N=20)
out = pd.DataFrame(
    np.array([l0, l2, linf, miss]).transpose(), columns=["l0", "l2", "linf", "miss"]
)
print(
    "Constrained l0:{} l2:{}  linf:{} miss: {} std: {}".format(
        l0.mean().item(),
        l2.mean().item(),
        linf.mean().item(),
        miss.mean().item(),
        miss.std().item(),
    )
)
print(
    "Constrained std l0:{} l2:{}  linf:{}".format(
        l0.std().item(), l2.std().item(), linf.std().item(),
    )
)
print(
    "suspicion: ",
    "l0:",
    (Xadv_s - X).norm(0).item(),
    "l2: ",
    (Xadv_s - X).norm(2).item(),
    "linf: ",
    (Xadv_s - X).norm(float("inf")).item(),
)


print("\n=================================================\n")

set_seed(4)
constrained = ConstrainedAdvPoisoningGlobal(
    delta=(Xadv_s - X).norm(float("inf")) / 2,
    s=1,
    clst_model=model,
    lb=1.0,
    G=20,
    mutation_rate=0.01,
    crossover_rate=0.85,
    zero_rate=0.1,
    domain_cons=[0, X.max() + 5],
    objective="AMI",
    mode="guided",
    link="centroids",
)
miss, l0, l2, linf = comparison(constrained, N=20)
out = pd.DataFrame(
    np.array([l0, l2, linf, miss]).transpose(), columns=["l0", "l2", "linf", "miss"]
)
print(
    "Constrained/2 l0:{} l2:{}  linf:{} miss: {} std: {}".format(
        l0.mean().item(),
        l2.mean().item(),
        linf.mean().item(),
        miss.mean().item(),
        miss.std().item(),
    )
)
print(
    "Constrained/2 std l0:{} l2:{}  linf:{}".format(
        l0.std().item(), l2.std().item(), linf.std().item(),
    )
)
print(
    "suspicion: ",
    "l0:",
    (Xadv_s - X).norm(0).item(),
    "l2: ",
    (Xadv_s - X).norm(2).item(),
    "linf",
    (Xadv_s - X).norm(float("inf")).item(),
)
