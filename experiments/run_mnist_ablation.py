import sys

sys.path.extend(["./"])
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import numpy as np
from clustorch.kmeans import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal
from experiments.run_fashion import filter_by_label
from experiments.utilities import ClusteringWrapper3Dto2D, set_seed
import ray
import csv
import datetime
from experiments.utilities import relabel

set_seed(4)


def write(results, params, i, writer):
    s, m_rate, c_rate, z_rate = params
    l0, l2, linf, miss_clustered, ami_adv, fitness = results
    writer.writerow(
        [i, s, m_rate, c_rate, z_rate]
        + [l0, l2, linf]
        + [miss_clustered, ami_adv, fitness]
    )


@ray.remote(num_cpus=1, num_gpus=1)
def test(X, Y, T, model):
    yhat = model.fit_predict(X)
    yhat = relabel(Y, yhat)
    Xadv_m, eps, ts_idx, d = T.forward(X, yhat, from_to=[1, 0])
    yadv_m = model.fit_predict(Xadv_m).cpu()
    yhat = yhat.cpu()
    ami_adv = adjusted_mutual_info_score(yhat, yadv_m)
    miss_clustered = min((yhat != yadv_m).sum(), (yhat != (1 - yadv_m)).sum()).item()
    l2 = (Xadv_m - X).norm(2).item()
    l0 = (Xadv_m - X).norm(0).item()
    linf = (Xadv_m - X).norm(float("inf")).item()
    fitness = T.get_fitness(Xadv_m, eps, yhat).item()
    return l0, l2, linf, miss_clustered, ami_adv, fitness


def run_on_ray(X, yhat, model, T, params, N=11, writer=None):
    ret_id = []
    for i in range(N):
        id_remote = test.remote(X, yhat, T, model)
        ret_id.append(id_remote)

    results = ray.get(ret_id)
    for i, ret in enumerate(results):
        write(ret, params, i, writer)


def run(X, yhat, model, s, delta_max, m_rate, c_rate, z_rate, n_iter=110, writer=None):
    T = ConstrainedAdvPoisoningGlobal(
        delta=delta_max,
        s=s,
        clst_model=model,
        lb=1.0 / 255,
        G=n_iter,
        mutation_rate=m_rate,
        crossover_rate=c_rate,
        zero_rate=z_rate,
        domain_cons=[0, 1],
        objective="AMI",
        mode="guided",
        link="centroids",
    )
    params = s, m_rate, c_rate, z_rate
    run_on_ray(X, yhat, model, T, N=11, writer=writer, params=params)


def study_parameters(file_name, X, yhat, model, s_lst, delta, n_iter=300):

    x_id = ray.put(X)
    y_id = ray.put(yhat)

    with open(file_name + ".csv", "w") as file:
        writer = csv.writer(file)
        header = "seed,s,m_rate,c_rate,z_rate," + "l0,l2,linf," + "miss,ami,fitness"

        writer.writerow([header])
        for s in s_lst:
            for m_rate in np.linspace(0.001, 0.201, 11):
                for c_rate in np.linspace(0.1, 1, 10):
                    for z_rate in np.linspace(0.001, 0.101, 11):
                        start = datetime.datetime.now()
                        run(
                            x_id,
                            y_id,
                            model,
                            s,
                            delta,
                            m_rate,
                            c_rate,
                            z_rate,
                            n_iter,
                            writer,
                        )
                        print(
                            datetime.datetime.now() - start, m_rate, c_rate, z_rate,
                        )
                        file.flush()
    file.close()


if __name__ == "__main__":
    ray.init()

    root = "./data/"
    trans = transforms.Compose([transforms.ToTensor(),])
    train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)

    torch.manual_seed(4)
    n_samples = 800

    X, Y = filter_by_label(
        x=train_set.data,
        y=train_set.targets,
        labels=[6, 9],  # [9,3]
        n_samples=n_samples,
        device="cuda:0",
    )
    k = len(Y.unique())
    kmeans = KMeans(n_clusters=k)
    model = ClusteringWrapper3Dto2D(kmeans)

    study_parameters(
        "ablation_mnist_all_d02_g300",
        X,
        Y,
        model,
        s_lst=[100, 200],
        delta=0.2,
        n_iter=400,
    )
