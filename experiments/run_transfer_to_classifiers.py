import sys

sys.path.extend(["./"])

import datetime, warnings
from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.metrics import adjusted_rand_score as ami
from sklearn.metrics import accuracy_score
from src.threat.clustering.constrained_poisoning import ConstrainedAdvPoisoningGlobal
from src.classification.rand_forest import RandomForestPartition
from src.classification.cw_deepnet import CWDeepNet
from src.classification.svm import SVMPartition
from clustorch.kmeans import KMeans
from experiments.utilities import *
from experiments.run_fashion import filter_by_label

from experiments.device import opts

DEVICE = opts.device
PATH = opts.path

PATH += "/pr2021/"


def eval_transferability(clf, Xadv, ts_idx, Y):
    acc = accuracy_score(clf.predict(Xadv[ts_idx]).cpu(), Y[ts_idx].cpu())
    return acc


def run(
    X,
    Y,
    model,
    clfs,
    seeds,
    dt_range,
    s_range,
    box=(0, 1),
    lb=1 / 255,
    outpath="transfer_fashion/",
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    n, m, k = X.shape
    yhat = model.fit_predict(X)
    yhat = relabel(Y, yhat)
    filename = model.to_string()

    with open(outpath + filename + ".csv", "w+") as writer:

        writer.write(
            "seed;model;delta;perc_samples;num_samples;totOverN;"
            "ARI;AMI;NMI;"
            "positive_eps_mean;negative_eps_mean;"
            "l0;l2;linf;"
            "ARI_hat;AMI_hat;NMI_hat;"
            "rand10_ACC;rand100_ACC;linear_SVM_ACC;rbf_SVM_ACC;cw_ACC"
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
                    Xadv, leps, ts_idx, direction = T.forward(X, yhat, from_to=[6, 9])
                    end = datetime.datetime.now()

                    yadv = model.fit_predict(Xadv)
                    ARI, AMI, NMI = attack_scores(Y, yadv)
                    ARI_hat, AMI_hat, NMI_hat = attack_scores(yhat, yadv)
                    l0, l2, linf = power_noise(X, Xadv)

                    eps = Xadv[ts_idx] - X[ts_idx]
                    peps = eps[eps > 0.0].mean()
                    neps = eps[eps > 0.0].mean()

                    clfs_acc = ""
                    for clf in clfs:
                        acc = eval_transferability(clf, Xadv, ts_idx, Y)
                        clfs_acc += ";" + str(acc)

                    n_points = len(ts_idx)
                    totOverN = len(ts_idx) / n
                    writer.write(
                        "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
                            seed,
                            model.to_string(),
                            dt,
                            s,
                            n_points,
                            totOverN,
                            ARI,
                            AMI,
                            NMI,
                            peps,
                            neps,
                            l0,
                            l2,
                            linf,
                            ARI_hat,
                            AMI_hat,
                            NMI_hat,
                        )
                        + clfs_acc
                    )
                    print("delta:{}  s:{}  ACC: ".format(dt, n_points) + clfs_acc)
                    print(
                        "======================== Time:",
                        end - start,
                        "=================================================",
                    )
                    writer.write("\n")
        writer.close()


root = "./data/"
trans = transforms.Compose([transforms.ToTensor(),])
# if not exist, download mnist dataset
train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.FashionMNIST(root=root, train=False, transform=trans, download=True)

torch.manual_seed(4)
n_samples = 800

dt_range = np.linspace(start=0.05, num=20, stop=1)
s_range = np.linspace(start=0.01, num=20, stop=0.6)

Xtrain, Ytrain = filter_by_label(
    x=train_set.data,
    y=train_set.targets,
    labels=range(10),
    encode=False,
    n_samples=None,
    device=DEVICE,
)

Xtest, Ytest = filter_by_label(
    x=test_set.data,
    y=test_set.targets,
    labels=[6, 9],
    encode=False,
    n_samples=None,
    device=DEVICE,
)


def train(Xtrain, Ytrain):
    rand_frst_10 = RandomForestPartition.from_train(
        (Xtrain.cpu(), Ytrain.cpu()), n_estimators=10
    )
    rand_frst_100 = RandomForestPartition.from_train(
        (Xtrain.cpu(), Ytrain.cpu()), n_estimators=100
    )
    linear_svm = SVMPartition.from_train((Xtrain.cpu(), Ytrain.cpu()), kernel="linear")
    rbf_svm = SVMPartition.from_train((Xtrain.cpu(), Ytrain.cpu()), kernel="rbf")
    return (rand_frst_10, rand_frst_100, linear_svm, rbf_svm)


def eval(clfs, Xtest, Ytest):
    for clf in clfs:
        print(clf.to_string(), " acc: ", clf.score(Xtest, Ytest))


def save(clfs, path):
    for clf in clfs:
        clf.store(path + clf.to_string() + ".clf")


def load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_linear = SVMPartition.load(path + "SVM_linear.clf", kernel="linear")
        svm_rbf = SVMPartition.load(path + "SVM_rbf.clf", kernel="rbf")
        rand_frst_10 = RandomForestPartition.load(
            path + "RandomForest_10.clf", n_estimators=10
        )
        rand_frst_100 = RandomForestPartition.load(
            path + "RandomForest_100.clf", n_estimators=100
        )
        cw_net = CWDeepNet.load(path + "cw_fashion.pt")
        cw_net.model = cw_net.model.to(DEVICE)
    return (rand_frst_10, rand_frst_100, svm_linear, svm_rbf, cw_net)


###################### TRAIN MODELS ###########################
clfs = train(Xtrain, Ytrain)
rand_frst_10, rand_frst_100, linear_svm, rbf_svm = clfs

eval(clfs, Xtest, Ytest)
save(clfs, path="./experiments/classifiers/")
###############################################################
clfs = load("./experiments/classifiers/")
eval(clfs, Xtest, Ytest)  # print accuracy target classifiers

k = len(Ytest.unique())
models = [KMeans(n_clusters=k)]
models = [ClusteringWrapper3Dto2D(m) for m in models]
seeds = [444, 314, 4, 410, 8089]
dt_ = np.linspace(start=0.001, num=20, stop=1)
s_ = np.linspace(start=0.1, num=6, stop=0.6)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for model in models:
        run(
            Xtest,
            Ytest,
            model,
            clfs,
            seeds,
            dt_range=dt_,
            s_range=s_,
            outpath=PATH + "transfer_fashion_69/",
        )
