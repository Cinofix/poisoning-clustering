import math
import random

import torch
from torch.utils.data import WeightedRandomSampler


class SingleGenOptimization:
    def __init__(
        self,
        G,  # number of generations
        delta,  # maximum amount of noise
        mutation_rate,
        crossover_rate,
        zero_rate,
        mode="bilateral",
        min_val=0.0,
        max_val=1.0,
        dtype=None,
        device="cuda:0",
    ):

        ## Genetic algorithm parameters
        self.G = G  # number of generations
        self.mutation_rate = mutation_rate  # pr of mutation
        self.crossover_rate = crossover_rate  # pr of crossover/reproduction
        self.zero_rate = zero_rate

        ## Evolution range
        self.delta = math.fabs(delta)  # we assume to be > 0
        self.mode = mode
        self.min_delta = -1.0 * self.delta if self.mode == "bilateral" else 0.0

        ## Domain constraints
        self.min_val = min_val
        self.max_val = max_val

        ## Device and type
        self.dtype = dtype
        self.device = device

    def rand_eps_mask(self, X, ts_idx):
        s = X.shape
        eps = torch.zeros(s, dtype=self.dtype, device=self.device)
        eps[ts_idx, :, :] = 1.0
        rand = (
            torch.rand(s, dtype=self.dtype, device=self.device)
            * (self.delta - self.min_delta)
            + self.min_delta
        )
        eps = eps * rand
        return eps

    def columns_swap_crossover(self, eps1, eps2, p=2):
        """
        Swap s columns of eps1 with s columns of eps2, where s is chosen randomly from [0, m//2]
        :param eps1:
        :param eps2:
        :return: New candidate noise
        """
        n, m, k = eps1.shape
        eps = eps1.clone()
        s = random.randint(0, m // 2)
        to_swap = random.sample(range(m), s)
        eps[:, to_swap, :] = eps2[:, to_swap, :]
        return eps

    def entries_swap_crossover(self, eps1, eps2):
        eps = eps1.clone()
        # flip for crossover of each entry
        flips = torch.rand(eps.shape) <= self.crossover_rate
        idx = torch.nonzero(flips, as_tuple=True)
        eps[idx] = eps2[idx]
        return eps

    def crossover(self, eps1, eps2):
        flip = torch.rand((1,), dtype=self.dtype, device=self.device)
        if flip <= self.crossover_rate:
            eps = self.entries_swap_crossover(eps1, eps2)
            return eps
        else:
            return eps1.clone()

    def add_noise(self, eps, v):
        tuple_v = v.split(1, dim=1)
        s = eps[tuple_v].shape
        eps[tuple_v] += torch.rand(s, dtype=self.dtype, device=self.device) * self.delta
        return torch.clamp_(eps, min=self.min_delta, max=self.delta)

    def change_noise(self, eps, v):
        tuple_v = v.split(1, dim=1)
        s = eps[tuple_v].shape
        eps[tuple_v] = (
            torch.rand(s, dtype=self.dtype, device=self.device)
            * (self.delta - self.min_delta)
            + self.min_delta
        )
        return eps

    def whom_to_mutate(self, eps, tg_idx, rate):
        n, m, k = eps.shape
        nt = len(tg_idx)
        # flip for mutation of each entry
        flips = torch.rand((nt, m, k)) < rate
        valid = torch.nonzero(flips)
        # set row index to target index
        valid[:, 0] = tg_idx[valid[:, 0]]
        return valid

    def set_to_zero(self, eps, v):
        tuple_v = v.split(1, dim=1)
        eps[tuple_v] = 0.0
        return eps

    def mutate(self, eps, tg_idx):
        ## Rand mutation
        v = self.whom_to_mutate(eps, tg_idx, self.mutation_rate)
        eps = self.add_noise(eps, v)
        ## Zero mutation
        v = self.whom_to_mutate(eps, tg_idx, self.zero_rate)
        eps = self.set_to_zero(eps, v)
        return eps

    def get_best_candidate(self, candidates):
        best_candidate = min(candidates, key=lambda x: candidates.get(x))
        return best_candidate

    def selection(self, candidates, mode="soft"):
        eps_candidates = [candidate.eps for candidate in candidates]
        scores = [candidate.score for candidate in candidates]
        scores = torch.tensor(scores, dtype=self.dtype, device=self.device)
        if mode == "min_max":
            scores = scores.clamp_(0.0, 1.0)
            weighted_probs = 1.0 - scores
        else:
            weighted_probs = torch.exp(-scores)
        weighted_probs = (weighted_probs + 1e-16) / (weighted_probs.sum() + 1e-16)
        i = list(WeightedRandomSampler(weighted_probs, num_samples=1))[0]
        candidate = eps_candidates[i]
        return candidate

    def evolve(self, eps1, tg_idx, candidates):
        eps2 = self.selection(candidates)
        eps_child = self.crossover(eps1, eps2)
        eps_child = self.mutate(eps_child, tg_idx)
        return eps_child

    def set_device(self, device):
        self.device = device

    def set_dtype(self, type):
        self.dtype = type
