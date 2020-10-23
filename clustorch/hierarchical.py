import torch
from clustorch.base import ClusteringModel
from sklearn.cluster import AgglomerativeClustering


class Hierarchical(ClusteringModel):
    """ Defines the  Hierarchical clustering  algorithm."""

    def __init__(
        self,
        n_clusters=2,
        affinity="euclidean",
        linkage="ward",
        verbose=False,
        seed=None,
    ):
        self.k = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.verbose = verbose
        self.status = {}

        self.model = AgglomerativeClustering(
            n_clusters=self.k, affinity=self.affinity, linkage=self.linkage
        )

        if isinstance(seed, int):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def fit(self, X, y=None, sample_weight=None):
        if X.is_cuda:
            y_hat = self.model.fit_predict(X.cpu().numpy())
        else:
            y_hat = self.model.fit_predict(X.numpy())
        return torch.from_numpy(y_hat)

    def to_string(self):
        return "Hierarchical " + self.affinity
