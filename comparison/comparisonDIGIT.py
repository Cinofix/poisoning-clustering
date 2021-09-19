import sys

sys.path.extend(["./"])

import torch, sys
import numpy as np
import pandas as pd
from clustorch.hierarchical import Hierarchical
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal
from experiments.utilities import ClusteringWrapper3Dto2D, set_seed
from comparison.utilities import comparison, show_fig
from optparse import OptionParser

op = OptionParser()
op.add_option(
    "--dataset",
    type=str,
    default="89",
    help="DIGIT dataset to analyze. Available: '89' or '14']",
)
(opts, args) = op.parse_args(sys.argv[1:])

X = np.load("comparison/DIGIT/X_org_{}.npy".format(opts.dataset))
Xadv_s = np.load("comparison/DIGIT/X_adv_{}.npy".format(opts.dataset))

X = torch.from_numpy(X).unsqueeze(2)
Xadv_s = torch.from_numpy(Xadv_s).unsqueeze(2)

eps_s = Xadv_s - X

h = Hierarchical(n_clusters=2)
model = ClusteringWrapper3Dto2D(h)
yhat = model.fit_predict(X)
yadv_s = model.fit_predict(Xadv_s)

set_seed(55)
T = ConstrainedAdvPoisoningGlobal(
    delta=(Xadv_s - X).norm(float("inf")) / 2,
    s=1,
    clst_model=model,
    lb=1.0,
    G=150,
    mutation_rate=0.2,
    crossover_rate=0.85,
    zero_rate=0.35,
    domain_cons=[0, 16],
    objective="AMI",
    mode="guided",
    link="centroids",
)
Xadv_m, leps, ts_idx, d = T.forward(X, yhat, from_to=[1, 0])
yadv_m = model.fit_predict(Xadv_m)
eps_m = Xadv_m - X
print("=================================================")
print("Ours miss:", min((yhat != yadv_m).sum(), (1 - yhat != yadv_m).sum()))
print("Suspicion Miss:", min((yhat != yadv_s).sum(), (1 - yhat != yadv_s).sum()))
print("=================================================\n")

idx_s = torch.nonzero(Xadv_s - X, as_tuple=False)[0, 0]
idx_m = torch.nonzero(Xadv_m - X, as_tuple=False)[0, 0]

show_fig(
    X,
    Xadv_s,
    Xadv_m,
    idx_m,
    idx_s,
    shape=(8, 8),
    out_file="digit_{}.png".format(opts.dataset),
)

set_seed(4)
unconstrained = ConstrainedAdvPoisoningGlobal(
    delta=16,
    s=1,
    clst_model=model,
    lb=1,
    G=150,
    mutation_rate=0.001,
    crossover_rate=0.85,
    zero_rate=0.25,
    domain_cons=[0, 16],
    objective="AMI",
    mode="guided",
    link="centroids",
)
miss, l0, l2, linf = comparison(X, yhat, model, unconstrained, N=20)
out = pd.DataFrame(
    np.array([l0, l2, linf, miss]).transpose(), columns=["l0", "l2", "linf", "miss"]
)
print(
    "Unconstrained l0:{} l2:{}  linf:{} miss: {} std: {}".format(
        l0.mean().item(),
        l2.mean().item(),
        linf.mean().item(),
        miss.mean().item(),
        miss.std().item(),
    )
)
print(
    "comparison: ",
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
    delta=(Xadv_s - X).norm(float("inf")),
    s=1,
    clst_model=model,
    lb=1.0,
    G=150,
    mutation_rate=0.01,
    crossover_rate=0.85,
    zero_rate=0.2,
    domain_cons=[0, 16],
    objective="AMI",
    mode="guided",
    link="centroids",
)
miss, l0, l2, linf = comparison(X, yhat, model, constrained, N=20)
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
    "comparison: ",
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
    G=150,
    mutation_rate=0.2,
    crossover_rate=0.85,
    zero_rate=0.35,
    domain_cons=[0, 16],
    objective="AMI",
    mode="guided",
    link="centroids",
)
miss, l0, l2, linf = comparison(X, yhat, model, constrained, N=20)
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
    "comparison: ",
    "l0:",
    (Xadv_s - X).norm(0).item(),
    "l2: ",
    (Xadv_s - X).norm(2).item(),
    "linf: ",
    (Xadv_s - X).norm(float("inf")).item(),
)
print("\n=================================================\n")
