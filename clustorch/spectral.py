import torch, math
from clustorch.base import ClusteringModel
from clustorch.kmeans import KMeans
from clustorch.metrics import pairwise


class SpectralClustering(ClusteringModel):
    def __init__(
        self,
        n_clusters=2,
        lmode="sym",
        similarity="gaussian",
        assign_labels="sq_kmeans",
        smode="knn",
        k=None,
        ensembling=None,
        kinit="kpp",
        kensembling=None,
        n_init=10,
        max_iter=300,
        max_tol=1e-3,
        diag=None,
        verbose=False,
        seed=None,
    ):
        self.n_clusters = n_clusters
        self.lmode = lmode
        self.similarity = similarity
        self.diag = diag
        self.smode = smode
        self.k = k
        self.ensembling = ensembling
        self.kensembling = kensembling
        self.kinit = kinit
        self.n_init = n_init
        self.max_iter = max_iter
        self.max_tol = max_tol
        self.verbose = verbose
        self.status = {}
        self.assign_labels = assign_labels

        # Init KMeans model
        squared = assign_labels == "sq_kmeans"

        self.model = KMeans(
            n_clusters=n_clusters,
            init="kpp",
            ensembling=None,
            verbose=verbose,
            seed=seed,
        )
        # Set random seeds
        if isinstance(seed, int):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def sigma_local_scaling(self, D, k=None, smode="knn"):
        """ Estimate the scaling factor sigma as suggested by
        Zelnik, Perona in Self-Tuning Spectral Clustering """
        n, m = D.shape
        if not isinstance(k, int):
            k = int(math.log10(n))
        s = D.sort(dim=1)[0]
        self.status["k"] = k if self.verbose else None
        if smode == "knn":
            return s[:, k]
        elif smode == "kavg":
            return s[:, :k].mean(dim=1)

    def affinity(self, X, similarity, diag=None):
        if similarity == "gaussian":
            scale = 2.0 * X.var() + 1e-16
            D = pairwise.dist_mat(X, squared=False)
            self.status["dist"] = D if self.verbose else None
            return pairwise.gaussian_affinity(D, scale, diag)
        elif similarity == "sq_gaussian":
            scale = 2.0 * X.var() + 1e-15
            D = pairwise.dist_mat(X, squared=True)
            self.status["dist"] = D if self.verbose else None
            return pairwise.gaussian_affinity(D, scale, diag)
        elif similarity == "gaussian_zp":  # gaussian similarity by Zelnik and Perona
            D = pairwise.dist_mat(X, squared=False)
            scale = self.sigma_local_scaling(D, self.k, self.smode)
            self.status["dist"] = D if self.verbose else None
            return pairwise.gaussian_affinity(D, scale, diag)
        elif similarity == "sq_gaussian_zp":  # gaussian similarity by Zelnik and Perona
            D = pairwise.dist_mat(X, squared=True)
            scale = self.sigma_local_scaling(D, self.k, self.smode)
            self.status["dist"] = D if self.verbose else None
            return pairwise.gaussian_affinity(D, scale, diag)
        elif similarity == "min_max":
            D = pairwise.dist_mat(X, squared=True)
            self.status["dist"] = D if self.verbose else None
            return pairwise.min_max_sim(D)
        elif similarity == "corr":
            return pairwise.correlation(X)
        elif similarity == "cosine":
            w1 = X.norm(p=2, dim=1, keepdim=True)
            return torch.mm(X, X.t()) / (w1 * w1.t())  # .clamp_(0,1)
            # return torch.mm(X, X.T)/(X.norm(p=2)**2.)
        elif similarity == "dot":
            return torch.mm(X, X.T)

    def laplacian(self, W):
        D = (W.sum(dim=1)) + 1e-16
        L = D.diag() - W
        if self.lmode == "unnorm":
            return L
        elif self.lmode == "sym":
            D = D ** -0.5
            return L * D.unsqueeze(0) * D.unsqueeze(1)
        elif self.lmode == "rw":
            return L * (1.0 / D).unsqueeze(1)

    def spectral_embedding(self, W):
        L = self.laplacian(W)
        _, eig_vectors = torch.symeig(L, eigenvectors=True)
        self.status["eig_v"] = eig_vectors if self.verbose else None
        self.status["laplacian"] = L if self.verbose else None
        U = eig_vectors[:, : self.n_clusters]
        if self.lmode == "rw":
            normalization = torch.norm(U, dim=1).unsqueeze(1) + 1e-16
            return U / normalization  # normalize rows to 1
        else:
            return U

    def fit(self, X):
        if self.ensembling == "rnd":
            rnd_spectral = RandSpectralClustering(
                self.n_clusters,
                self.n_init,
                self.lmode,
                self.similarity,
                self.assign_labels,
                self.smode,
                self.k,
                self.max_iter,
                self.max_tol,
                self.diag,
            )
            return rnd_spectral.fit(X)
        elif self.ensembling == "multiple_sim":
            sim_spectral = MultipleSimilaritySpectralClustering(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                lmode=self.lmode,
                assign_labels=self.assign_labels,
                smode=self.smode,
                k=self.k,
                max_iter=self.max_iter,
                max_tol=self.max_tol,
                diag=self.diag,
            )
            return sim_spectral.fit(X)

        W = self.affinity(X, self.similarity, self.diag)
        self.status["affinity"] = W if self.verbose else None
        T = self.spectral_embedding(W)
        if self.verbose:
            clst, km_status = self.model.fit(T)
        else:
            clst = self.model.fit(T)
        self.status["embedding"] = T if self.verbose else None
        self.status["similarity"] = W if self.verbose else None
        self.status["kmeans_status"] = km_status if self.verbose else None
        if self.verbose:
            return clst, self.status
        return clst

    def fit_from_affinity(self, W):  # precomputed affinity matrix
        T = self.spectral_embedding(W)
        clst, km_status = self.model.fit(T)
        self.status["embedding"] = T if self.verbose else None
        self.status["similarity"] = W if self.verbose else None
        self.status["kmeans_status"] = km_status if self.verbose else None
        return clst, self.status

    def to_string(self):
        return "Spectral"


from sklearn.metrics import silhouette_score
import operator


class RandSpectralClustering(ClusteringModel):
    def __init__(
        self,
        n_clusters=2,
        n_init=20,
        lmode="sym",
        similarity="gaussian",
        assign_labels="sq_kmeans",
        smode="knn",
        k=None,
        max_iter=300,
        max_tol=1e-3,
        diag=None,
        verbose=False,
        seed=None,
    ):
        self.spectral = SpectralClustering(
            n_clusters,
            lmode,
            similarity,
            assign_labels,
            smode,
            k,
            ensembling=None,
            kinit="rnd",
            n_init=n_init,
            max_iter=max_iter,
            max_tol=max_tol,
            diag=diag,
        )
        self.n_init = n_init

    def fit(self, X, y=None, sample_weight=None):
        yhat = self.spectral.fit_predict(X)
        return yhat

    def to_string(self):
        return "RandSpectralEnsembling"


class MultipleSimilaritySpectralClustering(ClusteringModel):
    def __init__(
        self,
        n_clusters=2,
        n_init=20,
        similarities=["corr", "min_max", "cosine"],
        lmode="sym",
        assign_labels="kmeans",
        smode="knn",
        k=None,
        max_iter=300,
        max_tol=1e-3,
        diag=None,
        verbose=False,
        seed=None,
    ):
        self.models = []
        for sim in similarities:
            self.models += [
                SpectralClustering(
                    n_clusters,
                    lmode,
                    sim,
                    assign_labels,
                    smode,
                    k,
                    ensembling=None,
                    kensembling="rnd",
                    kinit="rnd",
                    n_init=n_init,
                    max_iter=max_iter,
                    max_tol=max_tol,
                    diag=diag,
                )
            ]
        self.similarities = similarities
        self.n_init = n_init

    def fit(self, X, y=None, sample_weight=None):
        clst = {}
        for model in self.models:
            yhat = model.fit_predict(X)
            if X.is_cuda:
                score = silhouette_score(X.cpu(), yhat.cpu())
            else:
                score = silhouette_score(X, yhat)
            clst[yhat] = score
        return max(clst.items(), key=operator.itemgetter(1))[0]

    def to_string(self):
        return "MultipleSpectralEnsembling"
