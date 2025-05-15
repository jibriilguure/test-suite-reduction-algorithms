import numpy as np

from utils import fitness, is_full_coverage


def bool_cov_matrix(mat: np.ndarray) -> np.ndarray:
    """Boolean matrix for fast coverage checks."""
    return mat.astype(bool)


def repair_vectorised(sol: np.ndarray, cov_bool: np.ndarray) -> np.ndarray:
    """Greedy pruning using coverage counts (vectorised)."""
    s = sol.copy()
    covered = cov_bool[s].sum(axis=0)          # how many tests cover each mod
    for idx in np.where(s)[0][::-1]:           # iterate backwards
        if covered[cov_bool[idx]].min() > 1:   # every module still ≥1 cover
            s[idx] = False
            covered -= cov_bool[idx]
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

    # ------------ greedy seed ------------
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
        #  initial particles
        greedy = self.greedy_solution()
        greedy ^= (np.random.rand(self.nTests) < self.flip)   # flip ≈5 % bits
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
            # one random matrix reused for all comparisons this iter
            r1 = np.random.rand(self.p, 1)
            r2 = np.random.rand(self.p, 1)

            diff_p = np.logical_xor(best_p, particles).astype(float)
            diff_g = np.logical_xor(g_best,  particles).astype(float)

            velocities = w*velocities + c1*r1*diff_p + c2*r2*diff_g
            probs      = self.sigmoid(velocities)
            rand_mat   = np.random.rand(self.p, self.nTests)
            particles  = rand_mat < probs

            # ----- evaluate & personal bests (no repair yet) -----
            for i in range(self.p):
                if not is_full_coverage(particles[i], self.C_bool):
                    continue
                cur_fit = fitness(particles[i], self.C, self.T)
                if cur_fit < best_f[i]:
                    best_p[i] = particles[i]
                    best_f[i] = cur_fit

            # ----- update global best & repair exactly once ------
            new_g_idx = best_f.argmin()
            if best_f[new_g_idx] < g_fit:
                g_best_raw = best_p[new_g_idx]
                g_best = repair_vectorised(g_best_raw, self.C_bool)
                g_fit  = fitness(g_best, self.C, self.T)
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= self.pat:
                print(f"⏹️ pso: Early stopping at iteration {it+1}")
                break

        return g_best
