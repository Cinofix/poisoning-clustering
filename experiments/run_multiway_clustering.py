import sys

sys.path.extend(["./"])

from experiments.utilities import *
import datetime
import os, warnings
from tqdm import tqdm
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal


def run(
    X,
    Y,
    y_target,
    model,
    seeds,
    dt_range,
    s_range,
    lb=1 / 255,
    box=(0, 1),
    outpath="AdvMNISTPoisoning_seed/",
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    n, m, k = X.shape
    yhat = model.fit_predict(X)
    # we introduce relabel to ensure that all clustering algorithms will move the same clusters
    yhat = relabel(Y, yhat)

    filename = model.to_string()

    with open(outpath + filename + ".csv", "w+") as writer:

        writer.write(
            "model;delta;perc_samples;num_samples;totOverN;"
            "ARI;AMI;NMI;"
            "l0;l2;linf;"
            "peps;neps;"
            "ARI_hat;AMI_hat;NMI_hat;seed"
        )
        writer.write("\n")

        for seed in seeds:
            for dt in tqdm(dt_range, total=len(dt_range)):
                for s in s_range:
                    set_seed(seed)
                    start = datetime.datetime.now()
                    print(
                        "\n====================== Starting time:",
                        start,
                        "model:",
                        model.to_string(),
                        "==============",
                    )

                    T = ConstrainedAdvPoisoningGlobal(
                        delta=dt,
                        s=s,
                        clst_model=model,
                        lb=lb,
                        G=110,
                        domain_cons=box,
                        objective="AMI",
                        mode="guided",
                        link="centroids",
                        mutation_rate=0.05,
                        crossover_rate=0.85,
                        zero_rate=0.001,
                    )

                    Xadv, ts_idx = T.multi_forward(X, yhat, y_target=y_target)
                    end = datetime.datetime.now()

                    yadv = model.fit_predict(Xadv)
                    ARI, AMI, NMI = attack_scores(Y, yadv)
                    ARI_hat, AMI_hat, NMI_hat = attack_scores(yadv, yhat)
                    l0, l2, linf = power_noise(X, Xadv)

                    eps = Xadv[ts_idx] - X[ts_idx]
                    peps = eps[eps > 0.0].mean()
                    neps = eps[eps < 0.0].mean()

                    n_points = len(ts_idx)
                    totOverN = len(ts_idx) / n

                    writer.write(
                        "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
                            model.to_string(),
                            dt,
                            s,
                            n_points,
                            totOverN,
                            ARI,
                            AMI,
                            NMI,
                            l0,
                            l2,
                            linf,
                            peps,
                            neps,
                            ARI_hat,
                            AMI_hat,
                            NMI_hat,
                            seed,
                        )
                    )

                    print(
                        "delta:{}  s:{}  totOverN:{}  AMI_hat:{}  pmean_eps:{} l2:{};".format(
                            dt, s, totOverN, AMI_hat, peps, l2,
                        )
                    )
                    print(
                        "======================== Time:",
                        end - start,
                        "=================================================",
                    )
                    writer.write("\n")
        writer.close()


def test_robustness(
    X, Y, y_target, dt_range, s_range, models, path, lb=1 / 255, box=(0, 1)
):
    # deltas = np.linspace(start=0.05, num=20, stop=1)
    # n_samples = np.linspace(start=0.01, num=20, stop=0.6)
    seeds = [8089, 410, 4, 314, 44]  # [44, 314, 4, 410, 8089]
    models = [ClusteringWrapper3Dto2D(m) for m in models]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for model in models:
            run(
                X,
                Y,
                y_target,
                model,
                seeds,
                dt_range=dt_range,
                s_range=s_range,
                outpath=path,
                box=box,
                lb=lb,
            )
