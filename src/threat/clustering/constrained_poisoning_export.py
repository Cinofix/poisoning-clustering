from src.threat.clustering.constrained_poisoning import *


class ConstrainedAdvPoisoningGlobalExport(ConstrainedAdvPoisoningGlobal):
    def global_evolve_perturbation(self, X, ts_idx):
        Y = self.clst_model.fit_predict(X)
        eps = self.rand_eps_mask(X, ts_idx)
        candidates = []
        fit_by_g = [self.get_fitness(X, torch.zeros(X.shape), Y)]

        for g in range(self.G):  # for each generation g
            Xadv = self.inject_noise(X, ts_idx, eps)
            fit_miss = self.get_fitness(Xadv, eps, Y)
            heappush(candidates, HeapItem(fit_miss, eps))
            eps = self.evolve(eps, ts_idx, candidates)
            fit_by_g.append(nsmallest(1, candidates)[0].score)
        candidate = nsmallest(1, candidates)[0]
        opt_score, opt_eps = candidate.getattributes()
        return opt_eps, fit_by_g

    def generate_adv_noise(self, X, tg_idx):
        Xadv = X.clone()
        opt_eps, fit_by_g = self.global_evolve_perturbation(X, tg_idx)
        Xadv[tg_idx, :] = X[tg_idx, :] + self.direction * opt_eps[tg_idx, :]
        Xadv = self.fix_to_constraints(Xadv)
        return Xadv, opt_eps, fit_by_g

    def forward(self, X, Y, from_to=[1, 0]):
        """
        Manipulate entries in an adversarial way for reaching a desired miss-clustering
        :param X: pure/estimated training set
        :param Y: pure/estimated labelling for entries in X
        :param from_to: [origin cluster, destination cluster]
        :return: Xadv, poisoned dataset obtained from X
        """
        self.set_device(X.device)
        self.set_dtype(X.dtype)

        tg_sensitive_idx = self.get_sensitive_entities(X, Y, from_to, self.s)
        self.direction = self.get_likage_direction(X, Y, tg_sensitive_idx, from_to)
        Xadv, eps, fit_by_g = self.generate_adv_noise(X, tg_sensitive_idx)
        return Xadv, eps, tg_sensitive_idx, self.direction, fit_by_g
