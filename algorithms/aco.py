import numpy as np
import time
from utils import fitness

class ACO_Optimizer:
    def __init__(self, coverage_matrix, time_array,
                num_ants=5,
                num_iterations=10,
                evaporation_rate=0.3,
                patience=3,
                max_runtime=10,
                alpha = 1.0, beta = 2.0,
                 ):
        self.C  = coverage_matrix.astype(bool)
        self.T  = time_array
        self.nTests, self.nMods = self.C.shape

        self.num_ants  = num_ants
        self.iters     = num_iterations
        self.alpha     = alpha
        self.beta      = beta
        self.rho       = evaporation_rate
        self.patience  = patience
        self.max_run   = max_runtime

        self.pheromone = np.ones(self.nTests, dtype=float)
        self.best_sol  = None
        self.best_fit  = float('inf')

    # ---------- main run ----------
    def run(self):
        no_imp  = 0
        start   = time.time()

        C_uint8 = self.C.astype(np.uint8)

        for it in range(self.iters):
            if time.time() - start > self.max_run:
                print("⏹️ Timeout at iter", it+1); break

            sols, fits, improved = [], [], False

            for _ in range(self.num_ants):
                sol = np.zeros(self.nTests, dtype=bool)
                uncovered = np.ones(self.nMods, dtype=bool)   # modules left

                # --- construct one ant solution ---
                while uncovered.any():
                    # how many uncovered mods each test would give
                    counts = C_uint8.dot(uncovered.astype(np.uint8))
                    counts[sol] = 0              # exclude already‐picked tests
                    if counts.max() == 0: break  # no progress possible

                    heuristic = (counts + 1e-6) / (self.T + 0.1)
                    probs = (self.pheromone ** self.alpha) * (heuristic ** self.beta)
                    probs[sol] = 0
                    probs /= probs.sum()

                    idx = np.random.choice(self.nTests, p=probs)
                    sol[idx] = True
                    uncovered &= ~self.C[idx]     # remove covered modules

                if uncovered.any():               # coverage failed
                    continue
                fit = fitness(sol, self.C, self.T)
                sols.append(sol); fits.append(fit)

                if fit < self.best_fit:
                    self.best_fit, self.best_sol = fit, sol.copy()
                    improved = True

            # ---- pheromone update (elite 2 ants) ----
            self.pheromone *= (1 - self.rho)
            if fits:
                for idx in np.argsort(fits)[:2]:
                    self.pheromone[sols[idx]] += 1.0 / fits[idx]

            no_imp = 0 if improved else no_imp + 1
            if no_imp >= self.patience:
                print("⏹️ Early stop, no improvement at iter", it+1)
                break

        return self.best_sol
