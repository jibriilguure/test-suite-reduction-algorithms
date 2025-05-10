import numpy as np
import random
from utils import fitness

class PSO_Optimizer:
    def __init__(self, coverage_matrix, time_array, num_particles=50, num_iterations=100, early_stop_patience=10):
        self.coverage_matrix = coverage_matrix
        self.time_array = time_array
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.early_stop_patience = early_stop_patience
        self.num_tests = coverage_matrix.shape[0]
        self.num_modules = coverage_matrix.shape[1]

    def greedy_solution(self):
        solution = np.zeros(self.num_tests, dtype=bool)
        covered = set()
        for i in np.argsort(self.time_array):
            modules = set(np.where(self.coverage_matrix[i])[0])
            if not modules.issubset(covered):
                solution[i] = True
                covered.update(modules)
            if len(covered) == self.num_modules:
                break
        return solution

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def run(self):
        # Initialization
        particles = [np.random.choice([False, True], size=self.num_tests) for _ in range(self.num_particles - 1)]
        particles.insert(0, self.greedy_solution())  # Greedy seed
        velocities = [np.random.uniform(-1, 1, size=self.num_tests) for _ in range(self.num_particles)]

        best_particles = particles.copy()
        best_fitnesses = [fitness(p, self.coverage_matrix, self.time_array) for p in particles]
        global_best = best_particles[np.argmin(best_fitnesses)].copy()
        global_best_fitness = min(best_fitnesses)

        no_improvement = 0
        w, c1, c2 = 0.5, 1.5, 1.5  # PSO coefficients

        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                diff_personal = np.logical_xor(best_particles[i], particles[i]).astype(float)
                diff_global = np.logical_xor(global_best, particles[i]).astype(float)

                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * diff_personal +
                    c2 * r2 * diff_global
                )

                probs = self.sigmoid(velocities[i])
                particles[i] = np.random.rand(self.num_tests) < probs

                current_fitness = fitness(particles[i], self.coverage_matrix, self.time_array)

                if current_fitness < best_fitnesses[i]:
                    best_particles[i] = particles[i].copy()
                    best_fitnesses[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = current_fitness
                        no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.early_stop_patience:
                print(f"⏹️ Early stopping at iteration {iteration + 1}")
                break

        return global_best
