import torch
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import accuracy_score as AC


class AdversarialBase:
    def __init__(self, lb, domain_cons, box_cons, objective):
        self.objective = objective

        ## Fitness weights - Adversarial Score weights
        self.lb = lb  # perturbation power weight

        ## Box and Domain Constraints
        self.domain_cons = domain_cons
        self.box_cons = box_cons

    @staticmethod
    def get_components_of_cluster(X, Y, target_label):
        return X[Y == target_label]

    @staticmethod
    def get_centroid(cluster):
        return torch.mean(cluster, dim=0)

    @staticmethod
    def nearest_idxpoints(cluster1, centroid2, s):
        # returns a Tensor D containing the closest points of cl1 to the centroid c2
        D = torch.pairwise_distance(cluster1, centroid2).squeeze(1)
        target_idx = torch.argsort(D, descending=False)[:s]
        return target_idx

    def fix_to_constraints(self, Xadv):
        return torch.clamp_(Xadv, min=self.domain_cons[0], max=self.domain_cons[1])

    def one_hot(self, x):
        n = len(x)
        xh = torch.zeros((n, x.max() + 1), device=x.device)
        xh[torch.arange(n), x.long()] = 1.0
        return xh

    def eval_adv_efficiency(self, Y, Yadv):
        score = float("inf")
        if self.objective == "frobenius":
            y = self.one_hot(Y)
            yadv = self.one_hot(Yadv)
            score = -torch.norm(y @ torch.t(y) - yadv @ torch.t(yadv), p=2) / len(Y)
        elif self.objective == "AMI":
            score = max(AMI(Y.cpu(), Yadv.cpu()), 0.0)
        elif self.objective == "NMI":
            score = max(NMI(Y.cpu(), Yadv.cpu()), 0.0)
        elif self.objective == "accuracy":
            score = AC(Y.cpu(), Yadv.cpu())
        return score

    def eval_objective_cost(self, miss_clust, eps):
        # if miss_clust == 0.: return 0.
        n, m, k = eps.shape
        fit_miss = miss_clust + self.lb * (
            torch.norm(eps, p=0) * torch.norm(eps, p=float("inf"))
        ) / (n * m * k)
        return fit_miss
