import numpy as np
import torch, random, os
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


class ClusteringWrapper3Dto2D:
    def __init__(self, model):
        self.model = model

    def fit_predict(self, X):
        """
        Since X has shape nxmxk (with k=1 for MNIST) we use only the first two dimensions
        :param X:
        :return:
        """
        return self.model.fit_predict(X.squeeze(2))

    def to_string(self):
        return self.model.to_string()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def attack_scores(Y, Ypred):
    Y = Y.cpu()
    Ypred = Ypred.cpu()
    ARI, AMI, NMI = (
        adjusted_rand_score(Y, Ypred),
        adjusted_mutual_info_score(Y, Ypred),
        normalized_mutual_info_score(Y, Ypred),
    )
    return ARI, AMI, NMI


def power_noise(X, Xadv):
    l0 = torch.dist(X, Xadv, p=0)
    l2 = torch.dist(X, Xadv, p=2)
    linf = torch.dist(X, Xadv, p=float("inf"))
    return l0, l2, linf


def store_tensors(name, Xadv, Yadv):
    os.makedirs(os.path.dirname(name + "tensors/"))
    torch.save(Xadv, name + "tensors/Xadv.pt")
    torch.save(Yadv, name + "tensors/Yadv.pt")


def store_parameters(name, model, delta, n_samples):
    with open(name + "parameters.csv", "w+") as writer:
        writer.write("model;delta;n_samples\n")
        writer.write("{};{};{}".format(model, delta, n_samples))
        writer.write("\n")
    writer.close()


def samples_in_cluster(X, Y, lb):
    nc, mc, kc = X[Y == lb].shape
    return nc


def relabel(y, y_hat):
    k = len(y_hat.unique())
    y_hat_rl = y_hat.clone()
    for i in range(k):
        l = torch.mode(y[y_hat == i])[0]
        y_hat_rl[y_hat == i] = l
    return y_hat_rl


def shuffle_indexes(idxs):
    return idxs[torch.randperm(len(idxs))]
