import sys

sys.path.extend(["./"])

import datetime, warnings
from tqdm import tqdm
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal

from experiments.utilities import *


def inject_transfer(X_adv, X_transf, yhat_transf, ts_idx, from_to):
    idx_transf = torch.nonzero(yhat_transf == from_to[0], as_tuple=True)[0]
    idx_transf = shuffle_indexes(idx_transf)[: len(ts_idx)]
    X_adv_transf = X_transf.clone()
    X_adv_transf[idx_transf] = X_adv[ts_idx]
    return X_adv_transf


def run(
    X,
    Y,
    X_transf,
    Y_transf,
    model,
    seeds,
    dt_range,
    s_range,
    box=(0, 1),
    lb=1 / 255,
    measure="AMI",
    mutation_rate=0.05,
    outpath="AdvMNISTPoisoning_seed/",
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    n, m, k = X.shape
    yhat = model.fit_predict(X)
    yhat = relabel(Y, yhat)

    yhat_transf = model.fit_predict(X_transf)
    yhat_transf = relabel(Y_transf, yhat_transf)

    filename = model.to_string()
    from_to = [1, 0]
    with open(outpath + filename + ".csv", "w+") as writer:

        writer.write(
            "model;delta;perc_samples;num_samples;totOverN;"
            "ARI;AMI;NMI;"
            "ARI_transf;AMI_transf;NMI_transf;"
            "l0;l2;linf;"
            "peps;neps;"
            "ARI_hat;AMI_hat;NMI_hat;"
            "ARI_hat_transf;AMI_hat_transf;NMI_hat_transf;"
            "ARI_hat_transf_white;AMI_hat_transf_white;NMI_hat_transf_white;seed"
        )
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

                    T = ConstrainedAdvPoisoningGlobal(
                        delta=dt,
                        s=s,
                        clst_model=model,
                        lb=lb,
                        G=110,
                        domain_cons=box,
                        objective=measure,
                        mode="guided",
                        link="centroids",
                        mutation_rate=mutation_rate,
                        crossover_rate=0.85,
                        zero_rate=0.001,
                    )
                    Xadv, leps, ts_idx, direction = T.forward(X, yhat, from_to)
                    end = datetime.datetime.now()
                    yadv = model.fit_predict(Xadv)
                    ARI, AMI, NMI = attack_scores(Y, yadv)
                    ARI_hat, AMI_hat, NMI_hat = attack_scores(yhat, yadv)
                    l0, l2, linf = power_noise(X, Xadv)

                    eps = Xadv[ts_idx] - X[ts_idx]
                    peps = eps[eps > 0.0].mean()
                    neps = eps[eps < 0.0].mean()

                    n_points = len(ts_idx)
                    totOverN = len(ts_idx) / n

                    Xadv_transf = inject_transfer(
                        Xadv, X_transf, yhat_transf, ts_idx, from_to
                    )
                    yadv_transf = model.fit_predict(Xadv_transf)
                    ARI_transf, AMI_transf, NMI_transf = attack_scores(
                        Y_transf, yadv_transf
                    )
                    ARI_hat_transf, AMI_hat_transf, NMI_hat_transf = attack_scores(
                        yhat_transf, yadv_transf
                    )
                    print(X_transf.shape, X.shape)

                    T = ConstrainedAdvPoisoningGlobal(
                        delta=dt,
                        s=s,
                        clst_model=model,
                        lb=lb,
                        G=110,
                        domain_cons=box,
                        objective=measure,
                        mode="guided",
                        link="centroids",
                        mutation_rate=mutation_rate,
                        crossover_rate=0.85,
                        zero_rate=0.001,
                    )
                    Xadv_transf_white, _, _, _ = T.forward(
                        X_transf, yhat_transf, from_to
                    )
                    end = datetime.datetime.now()
                    yadv_transf_white = model.fit_predict(Xadv_transf_white)
                    (
                        ARI_hat_transf_white,
                        AMI_hat_transf_white,
                        NMI_hat_transf_white,
                    ) = attack_scores(yhat_transf, yadv_transf_white)

                    writer.write(
                        "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
                            model.to_string(),
                            dt,
                            s,
                            n_points,
                            totOverN,
                            ARI,
                            AMI,
                            NMI,
                            ARI_transf,
                            AMI_transf,
                            NMI_transf,
                            l0,
                            l2,
                            linf,
                            peps,
                            neps,
                            ARI_hat,
                            AMI_hat,
                            NMI_hat,
                            ARI_hat_transf,
                            AMI_hat_transf,
                            NMI_hat_transf,
                            ARI_hat_transf_white,
                            AMI_hat_transf_white,
                            NMI_hat_transf_white,
                            seed,
                        )
                    )

                    print(
                        "delta:{}  s:{}  AMI_hat:{} AMI_hat_transf:{} pmean_eps:{} l2:{};".format(
                            dt, s, AMI_hat, AMI_hat_transf, peps, l2
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
    X,
    Y,
    X_transf,
    Y_transf,
    dt_range,
    s_range,
    models,
    path,
    lb=1 / 255,
    box=(0, 1),
    measure="AMI",
    mutation_rate=0.05,
):
    # deltas = np.linspace(start=0.05, num=20, stop=1)
    # n_samples = np.linspace(start=0.01, num=20, stop=0.6)
    seeds = [444, 314, 4, 410, 8089]
    models = [ClusteringWrapper3Dto2D(m) for m in models]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for model in models:
            run(
                X,
                Y,
                X_transf,
                Y_transf,
                model,
                seeds,
                dt_range=dt_range,
                s_range=s_range,
                lb=lb,
                outpath=path,
                box=box,
                measure=measure,
                mutation_rate=mutation_rate,
            )
