import numpy as np
import time
from utils import fitness
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

class ACO_Optimizer:
    def __init__(self, coverage_matrix, time_array,
                 num_ants=5, num_iterations=10,
                 evaporation_rate=0.3, patience=3,
                 max_runtime=10, alpha=1.0, beta=2.0):

        self.C = csr_matrix(coverage_matrix)  # sparse matrix
        self.T = time_array
        self.nTests, self.nMods = self.C.shape

        self.num_ants = num_ants
        self.iters = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.patience = patience
        self.max_run = max_runtime

        self.pheromone = np.ones(self.nTests, dtype=float)
        self.best_sol = None
        self.best_fit = float('inf')

    def greedy_fallback(self):
        sol = np.zeros(self.nTests, dtype=bool)
        covered = set()
        for i in np.argsort(self.T):
            mods = set(self.C[i].indices)
            if not mods.issubset(covered):
                sol[i] = True
                covered.update(mods)
            if len(covered) == self.nMods:
                break
        return sol

    def construct_ant(self):
        sol = np.zeros(self.nTests, dtype=bool)
        uncovered = np.ones(self.nMods, dtype=bool)
        loop_guard = 0
        max_steps = int(self.nTests * 1.5)

        while uncovered.any() and loop_guard < max_steps:
            loop_guard += 1
            uncovered_mods = np.flatnonzero(uncovered)

            # Count coverage per test
            counts = self.C[:, uncovered_mods].sum(axis=1).A1
            counts[sol] = 0
            if counts.max() == 0:
                break

            heuristic = (counts + 1e-6) / (self.T + 0.1)
            probs = (self.pheromone ** self.alpha) * (heuristic ** self.beta)
            probs[sol] = 0

            total = probs.sum()
            if total == 0:
                break
            probs /= total

            idx = np.random.choice(self.nTests, p=probs)
            sol[idx] = True
            uncovered[self.C[idx].indices] = False

            if uncovered.sum() <= 5:
                for j in np.argsort(self.T):
                    if not sol[j]:
                        indices = self.C[j].indices
                        if np.any(uncovered[indices]):
                            sol[j] = True
                            uncovered[indices] = False
                    if not uncovered.any():
                        break
                break

        if uncovered.any():
            return None, float('inf')
        fit = fitness(sol, self.C.toarray(), self.T)
        return sol, fit

    def run(self):
        no_imp = 0
        start = time.time()

        for it in range(self.iters):
            if time.time() - start > self.max_run:
                print("⏹️ Timeout at iter", it + 1)
                break

            # Parallelize ant construction
            results = Parallel(n_jobs=-1)(delayed(self.construct_ant)() for _ in range(self.num_ants))
            sols, fits = zip(*[r for r in results if r[0] is not None])

            if not sols:
                continue

            improved = False
            for sol, fit in zip(sols, fits):
                if fit < self.best_fit:
                    self.best_fit = fit
                    self.best_sol = sol
                    improved = True

            # Update pheromone
            self.pheromone *= (1 - self.rho)
            top_k = np.argpartition(fits, 2)[:2]
            for idx in top_k:
                self.pheromone[sols[idx]] += 1.0 / (fits[idx] + 1e-6)

            no_imp = 0 if improved else no_imp + 1
            if no_imp >= self.patience:
                print("ACO: Early stop, no improvement at iter", it + 1)
                break

        if self.best_sol is None:
            print("ACO failed. Using greedy fallback.")
            return self.greedy_fallback()

        return self.best_sol
