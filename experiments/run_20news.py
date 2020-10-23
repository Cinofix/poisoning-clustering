from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from run_experiments import test_robustness
from clustorch.kmeans import KMeans
from clustorch.spectral import SpectralClustering
import torch
from experiments.device import opts

DEVICE = opts.device
PATH = opts.path

PATH += "/pr2021/"


def shuffle_indexes(idxs):
    return idxs[torch.randperm(len(idxs))]


def get_20news_dataset(categories, opts, device, n=1400):

    dataset = fetch_20newsgroups(
        subset="all", categories=categories, shuffle=True, random_state=42
    )

    labels = dataset.target
    vectorizer = TfidfVectorizer(
        # max_df=0.5,
        max_features=opts["n_features"],
        # min_df=2,
        stop_words="english",
        use_idf=True,
    )
    X = vectorizer.fit_transform(dataset.data)
    svd = TruncatedSVD(opts["n_components"])
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)

    x = torch.from_numpy(X).unsqueeze(2).to(device)
    y = torch.from_numpy(labels).to(device)

    idx = shuffle_indexes(torch.arange(x.shape[0]))
    x = x[idx][:n]
    y = y[idx][:n]
    return x, y


def main():

    torch.manual_seed(4)

    dt_range = np.linspace(start=0.001, num=15, stop=0.3)
    s_range = np.linspace(start=0.01, num=15, stop=0.3)
    opts = {"n_features": 10000, "n_components": 80}
    categories = ["rec.sport.baseball", "talk.politics.guns"]
    X, Y = get_20news_dataset(categories=categories, opts=opts, device=DEVICE, n=1400)
    k = len(Y.unique())
    models = [
        KMeans(n_clusters=k, verbose=False, init="rnd", n_init=50, squared=True),
        SpectralClustering(n_clusters=k, ensembling="multiple_sim", lmode="rw"),
    ]
    test_robustness(
        X,
        Y,
        models=models,
        dt_range=dt_range,
        s_range=s_range,
        lb=1 / 255,
        path=PATH + "/20news_ball_gun/",
        box=(-5, 5),
    )


if __name__ == "__main__":
    main()
