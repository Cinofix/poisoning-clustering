import sys

sys.path.extend(["./"])

import numpy as np
import torch
from comparison.utilities import SKlearnClusteringWrapper
from sklearn.cluster import KMeans
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal
from experiments.utilities import set_seed


X = np.load("comparison/HAND-3-5/kme_X_org.npy")
Xadv_s = np.load("comparison/HAND-3-5/kme_X_adv.npy")

X = torch.from_numpy(X).unsqueeze(2)
Xadv_s = torch.from_numpy(Xadv_s).unsqueeze(2)
eps_s = Xadv_s - X

kmeans = KMeans(n_clusters=2, random_state=42)
model = SKlearnClusteringWrapper(kmeans)

y = torch.from_numpy(np.genfromtxt("comparison/HAND-3-5/lo.csv", delimiter="\n"))

yadv_s = model.fit_predict(Xadv_s)
yhat = model.fit_predict(X)

print("Suspicion Miss:", min((yhat != yadv_s).sum(), (1 - yhat != yadv_s).sum()))

set_seed(4)
T = ConstrainedAdvPoisoningGlobal(
    delta=(Xadv_s - X).norm(float("inf")),
    s=1,
    clst_model=model,
    lb=0.0,
    G=10,
    mutation_rate=0.02,
    crossover_rate=0.85,
    zero_rate=0.0,
    domain_cons=[X.min() - 50, X.max() + 50],
    objective="AMI",
    mode="guided",
    link="centroids",
)
Xadv_m, leps, ts_idx, d = T.forward(X, yhat, from_to=[1, 0])
yadv_m = model.fit_predict(Xadv_m)
eps_m = Xadv_m - X
print("\n=================================================\n")
print("Ours miss:", min((yhat != yadv_m).sum(), (1 - yhat != yadv_m).sum()))
print("Suspicion Miss:", min((yhat != yadv_s).sum(), (1 - yhat != yadv_s).sum()))

print("suspicion: ", "l2:", (Xadv_s - X).norm(2), "l0: ", (Xadv_s - X).norm(0))
print("ours: ", "l2:", (Xadv_m - X).norm(2), "l0: ", (Xadv_m - X).norm(0))
print("\n=================================================\n")

idx_s = torch.nonzero(Xadv_s - X, as_tuple=False)[0, 0]
idx_m = torch.nonzero(Xadv_m - X, as_tuple=False)[0, 0]
print("target sample", idx_s, idx_m)

import pandas as pd


def comparison(T, N=20):
    miss_clustered = np.zeros(N)
    l2 = np.zeros(N)
    linf = np.zeros(N)
    l0 = np.zeros(N)
    for i in range(N):
        Xadv_m, leps, ts_idx, d = T.forward(X, yhat, from_to=[1, 0])
        if ts_idx != idx_s:
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
    G=50,
    mutation_rate=0.015,
    crossover_rate=0.85,
    zero_rate=0.20,
    domain_cons=[-float("inf"), float("inf")],
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
    lb=1,
    G=50,
    # mutation_rate=0.15,
    mutation_rate=0.015,
    # crossover_rate=0.20,
    crossover_rate=0.85,
    zero_rate=0.20,
    domain_cons=[-float("inf"), float("inf")],
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
