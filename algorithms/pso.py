import numpy as np
from utils import fitness, is_full_coverage

def bool_cov_matrix(mat: np.ndarray) -> np.ndarray:
    return mat.astype(bool)

def repair_vectorised(sol: np.ndarray, cov_bool: np.ndarray) -> np.ndarray:
    s = sol.copy()
    covered = cov_bool[s].sum(axis=0)
    for idx in np.where(s)[0][::-1]:
        if covered[cov_bool[idx]].min() > 1:
            s[idx] = False
            covered -= cov_bool[idx]
    return s

def repair_full_coverage(sol: np.ndarray, cov_bool: np.ndarray) -> np.ndarray:
    """Add missing modules using a greedy fix."""
    s = sol.copy()
    covered = cov_bool[s].any(axis=0)

    if covered.all():
        return s

    missing = ~covered
    for i in np.argsort(cov_bool.sum(axis=1)):  # greedy: low-coverage tests first
        if not s[i] and cov_bool[i][missing].any():
            s[i] = True
            covered |= cov_bool[i]
            missing = ~covered
            if covered.all():
                break
    return s

class PSO_Optimizer:
    def __init__(self, coverage_matrix, time_array,
                 num_particles=5, num_iterations=10,
                 early_stop_patience=3, flip_rate=0.05):
        self.C       = coverage_matrix
        self.C_bool  = bool_cov_matrix(coverage_matrix)
        self.T       = time_array
        self.p       = num_particles
        self.iters   = num_iterations
        self.pat     = early_stop_patience
        self.flip    = flip_rate

        self.nTests, self.nMods = self.C.shape

    def greedy_solution(self):
        sol = np.zeros(self.nTests, dtype=bool)
        covered = np.zeros(self.nMods, dtype=bool)
        for i in np.argsort(self.T):
            if covered.all(): break
            if not (covered & self.C_bool[i]).all():
                sol[i] = True
                covered |= self.C_bool[i]
        return sol

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))

    def run(self):
        greedy = self.greedy_solution()
        greedy ^= (np.random.rand(self.nTests) < self.flip)
        greedy = repair_vectorised(greedy, self.C_bool)

        particles  = np.random.rand(self.p, self.nTests) < 0.5
        particles[0] = greedy
        velocities = np.random.uniform(-1, 1, size=(self.p, self.nTests))

        best_p = particles.copy()
        best_f = np.array([fitness(x, self.C, self.T) for x in particles])
        g_idx  = best_f.argmin()
        g_best = best_p[g_idx].copy()
        g_fit  = best_f[g_idx]

        no_imp = 0
        w, c1, c2 = 0.7, 1.5, 1.0

        for it in range(self.iters):
            r1 = np.random.rand(self.p, 1)
            r2 = np.random.rand(self.p, 1)

            diff_p = np.logical_xor(best_p, particles).astype(float)
            diff_g = np.logical_xor(g_best,  particles).astype(float)

            velocities = w * velocities + c1 * r1 * diff_p + c2 * r2 * diff_g
            probs      = self.sigmoid(velocities)
            rand_mat   = np.random.rand(self.p, self.nTests)
            particles  = rand_mat < probs

            for i in range(self.p):
                if not is_full_coverage(particles[i], self.C_bool):
                    continue
                cur_fit = fitness(particles[i], self.C, self.T)
                if cur_fit < best_f[i]:
                    best_p[i] = particles[i]
                    best_f[i] = cur_fit

            new_g_idx = best_f.argmin()
            if best_f[new_g_idx] < g_fit:
                g_best = best_p[new_g_idx].copy()
                g_fit  = best_f[new_g_idx]
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= self.pat:
                print(f"pso: Early stopping at iteration {it+1}")
                break

        # Final fix: Ensure full coverage using greedy repair
        if not is_full_coverage(g_best, self.C_bool):
            g_best = repair_full_coverage(g_best, self.C_bool)

        return g_best
