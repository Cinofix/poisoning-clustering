import sys

sys.path.extend(["./"])

import datetime, warnings
from tqdm import tqdm
from src.threat.clustering.constrained_poisoning_export import (
    ConstrainedAdvPoisoningGlobalExport,
)
from experiments.utilities import *


def run(
    X,
    Y,
    model,
    seeds,
    dt_range,
    s_range,
    box=(0, 1),
    lb=1 / 255,
    outpath="ExportAdvMNISTPoisoning_seed/",
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    n, m, k = X.shape
    yhat = model.fit_predict(X)
    # we introduce relabel to ensure that all clustering algorithms will move the same clusters
    yhat = relabel(Y, yhat)
    filename = model.to_string()

    with open(outpath + filename + ".csv", "w+") as writer:

        writer.write("model;delta;perc_samples;num_samples;totOverN;" "g;fit;seed")
        writer.write("\n")
        for seed in seeds:
            for dt in tqdm(dt_range, total=len(dt_range)):
                for s in s_range:
                    set_seed(seed)
                    # n_points = int(nc * s)
                    start = datetime.datetime.now()
                    print(
                        "\n====================== Starting time:",
                        start,
                        "model:",
                        model.to_string(),
                        "==============",
                    )

                    T = ConstrainedAdvPoisoningGlobalExport(
                        delta=dt,
                        s=s,
                        clst_model=model,
                        lb=lb,
                        G=150,
                        domain_cons=box,
                        objective="AMI",
                        mode="guided",
                        link="centroids",
                        mutation_rate=0.05,
                        crossover_rate=0.85,
                        zero_rate=0.001,
                    )

                    Xadv, leps, ts_idx, direction, fit_by_g = T.forward(X, yhat)
                    end = datetime.datetime.now()

                    yadv = model.fit_predict(Xadv)
                    ARI_hat, AMI_hat, NMI_hat = attack_scores(yhat, yadv)

                    n_points = len(ts_idx)
                    totOverN = len(ts_idx) / n

                    for i, fit in enumerate(fit_by_g):

                        writer.write(
                            "{};{};{};{};{};{};{};{}".format(
                                model.to_string(),
                                dt,
                                s,
                                n_points,
                                totOverN,
                                i,
                                fit,
                                seed,
                            )
                        )
                        writer.write("\n")

                    print("delta:{}  s:{}  AMI_hat:{}".format(dt, s, AMI_hat))
                    print(
                        "======================== Time:",
                        end - start,
                        "=================================================",
                    )
        writer.close()


def test_robustness(X, Y, dt_range, s_range, models, path, lb=1 / 255, box=(0, 1)):
    # deltas = np.linspace(start=0.05, num=20, stop=1)
    # n_samples = np.linspace(start=0.01, num=20, stop=0.6)
    seeds = [444, 314, 4, 410, 8089]  # [2, 4, 6, 8, 10]
    models = [ClusteringWrapper3Dto2D(m) for m in models]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for model in models:
            run(
                X,
                Y,
                model,
                seeds,
                dt_range=dt_range,
                s_range=s_range,
                lb=lb,
                outpath=path,
                box=box,
            )
