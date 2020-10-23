class ClusteringModel:
    """ ClusteringModel defines the base for the clustering models hierarchy."""

    def fit_predict(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : pytorch tensor of shape [n_samples, n_features]
            Training set.
        y : pytorch tensotr of shape [n_samples]
            Target values.
        Returns
        -------
        X_out : pytorch tensor of shape [n_samples, n_features_new] containing cluster labels
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params)
