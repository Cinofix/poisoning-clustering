import torch


def dist_mat(X, squared=True):
    D = torch.cdist(X, X)
    return D if not squared else D.sqrt()


def gaussian_affinity(D, scale, diag=None):
    scale = scale + 1e-15
    W = torch.exp(-D / scale)
    if isinstance(diag, float):
        range_ = torch.arange(W.shape[0])
        W[range_, range_] = diag
    return W


def min_max_sim(D):
    return D.max() - D + D.min()


def correlation(X):
    X = X - X.mean(dim=0)
    norms = torch.norm(X, dim=1).view(-1, 1)
    W = torch.mm(X, X.t()) / (norms * norms.t() + 1e-16)
    return torch.clamp(W, min=0.0, max=1.0)


def sim2distance(W, method="max"):
    if method == "max":
        return 1.0 - W
    elif method == "euclidean":
        return dist_mat(W, squared=False)
    elif method == "sq_euclidean":
        return dist_mat(W, squared=True)
