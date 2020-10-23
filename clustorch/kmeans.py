import torch
from clustorch.base import ClusteringModel
from torch.utils.data import WeightedRandomSampler
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KMeans(ClusteringModel):
    """ Defines the  K-Means clustering  algorithm."""

    # TODO: ADD PRECOMPUTED DISTANCE MATRIX
    def __init__(
        self,
        n_clusters=2,
        init="kpp",
        n_init=10,
        ensembling=None,
        max_iter=300,
        max_tol=1e-3,
        squared=True,
        verbose=False,
        seed=None,
    ):
        self.k = n_clusters
        self.init = init
        self.ensembling = ensembling
        self.n_init = n_init
        self.max_iter = max_iter
        self.max_tol = max_tol
        self.squared = squared
        self.verbose = verbose
        self.status = {}
        if isinstance(seed, int):
            set_seed(seed)
            # torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def init_kpp(self, X):
        """ KMeans++ is another way for choosing clusters' centroids
        1. Choose one center uniformly at random from among the data points.
        2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
        4. Repeat Steps 2 and 3 until k centers have been chosen."""
        n, m = X.shape
        centroids = torch.zeros((self.k, m), dtype=X.dtype, device=X.device)
        centroids[0, :] = X[torch.randperm(n)][0, :]  # 1

        for i in range(1, self.k):
            D = ((X - centroids[i - 1, :]) ** 2.0).sum(dim=1)  # 2
            weighted_prob = D / (D.sum() + 1e-16)
            candidate = list(WeightedRandomSampler(weighted_prob, num_samples=1))[0]
            centroids[i, :] = X[candidate, :]
        return centroids

    def eval(self, X, C):
        D = self.eucl_distances(X, C, self.squared)
        labels = self.assign_labels(D)

    def init_rnd(self, X):
        # Initializes clusters as k randomly selected points from points.
        r = torch.randperm(X.shape[0])
        return X[r][: self.k]

    def initialize_clusters(self, X):
        if self.init == "rnd":
            return self.init_rnd(X)
        elif self.init == "kpp":  # KMeans++
            return self.init_kpp(X)

    # Function for calculating the distance between centroids
    def eucl_distances(self, X, C, squared=True):
        """ Returns the distance the centroid is from each data point in points.
        """
        # X n,m     --> 1,n,m
        # C p,m     --> p,1,m
        # the result will be a p,n,m
        D = torch.cdist(X.unsqueeze(0).float(), C.unsqueeze(1).float())[:, :, 0].pow(2)
        # D = ((X.unsqueeze(0) - C.unsqueeze(1)) ** 2.).sum(dim=2)
        return D if not squared else D.sqrt()
        # ((X.unsqueeze(0) - C.unsqueeze(1)) ** 2.).sum(dim=2)  # ==> pxn
        # (X.unsqueeze(0)-C.unsqueeze(1)) broadcasting facendo combaciare le
        # m ==> ottengo una pxnxm che contiene le distanze per ogni componente m
        # **2) elevo al quadrato le distanze
        # .sum(dim=2) somma delle distanze al quadrato (x-xo)^2 + (y-yo)^2

    def assign_labels(self, D):
        return D.argmin(dim=0)

    def update_centroids(self, X, labels):
        n, m = X.shape
        C = torch.zeros((self.k, m), dtype=X.dtype, device=X.device)

        for i in range(self.k):
            C[i] = torch.mean(X[torch.nonzero(labels == i), :], dim=0)
        return C

    def loss(self, C, C_pred):
        # sum( C_i - Cpred_i , dim=1)^2  scarti per feature
        # sum(sum( C_i - Cpred_i )^2)  scarto totale
        return torch.sum(torch.sum((C - C_pred) ** 2.0, dim=1))

    def fit(self, X, y=None, sample_weight=None):
        n, m = X.shape
        i, tol = 0, float("inf")  # iteration step and centroids variance tolerance
        if torch.all(torch.eq(X, X[0])):  # if all elements are equals
            return torch.zeros(n, dtype=torch.long, device=X.device)

        if self.ensembling == "rnd":
            rnd_model = RandKMeans(
                n_clusters=self.k,
                n_init=self.n_init,
                max_iter=self.max_iter,
                max_tol=self.max_tol,
                squared=self.squared,
                verbose=False,
            )
            return rnd_model.fit(X)

        C = self.initialize_clusters(X)  # init centroids for clusters

        if self.verbose:
            self.status["centroids"] = [C]
            self.status["loss"] = []
            # only for storing the first random partitions
            D = self.eucl_distances(X, C, self.squared)
            labels = self.assign_labels(D)
            self.status["labels"] = [labels]

        while i < self.max_iter and tol > self.max_tol:
            C_pred = C
            D = self.eucl_distances(X, C, self.squared)
            labels = self.assign_labels(D)
            C = self.update_centroids(X, labels)
            tol = self.loss(C, C_pred)
            i = i + 1

            if self.verbose:
                self.status["centroids"].append(C)
                self.status["labels"].append(labels)
                self.status["loss"].append(tol)

        self.status["D"] = D
        self.status["iter"] = i

        if self.verbose:
            return labels, self.status
        return labels

    def to_string(self):
        return "KMeans"


from sklearn.metrics import silhouette_score
import operator


class RandKMeans(ClusteringModel):

    # TODO: ADD PRECOMPUTED DISTANCE MATRIX
    def __init__(
        self,
        n_clusters=2,
        n_init=100,
        max_iter=300,
        max_tol=1e-3,
        squared=True,
        verbose=False,
        seed=None,
    ):

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="kpp",
            n_init=0,
            ensembling=None,
            max_iter=max_iter,
            max_tol=max_tol,
            squared=squared,
        )
        self.n_init = n_init

    def fit(self, X, y=None, sample_weight=None):
        clst = {}
        x = X.clone().cpu()
        try:
            for i in range(self.n_init):
                yhat = self.kmeans.fit_predict(X)
                if X.is_cuda:
                    score = silhouette_score(x, yhat.cpu())
                else:
                    score = silhouette_score(X, yhat)
                clst[yhat] = score
            return max(clst.items(), key=operator.itemgetter(1))[0]
        except:
            return self.kmeans.fit_predict(X)
