import random

import numpy as np

from utils import fitness


class PSO_Optimizer:
    def __init__(self, coverage_matrix, time_array, num_particles=50, num_iterations=100):
        self.coverage_matrix = coverage_matrix
        self.time_array = time_array
        self.num_particles = num_particles
        self.num_iterations = num_iterations

    def run(self):
        particles = [np.random.choice([False, True], size=len(self.coverage_matrix)) for _ in range(self.num_particles)]
        velocities = [np.random.uniform(-1, 1, size=len(self.coverage_matrix)) for _ in range(self.num_particles)]
        best_particles = particles.copy()
        best_fitnesses = [fitness(p, self.coverage_matrix, self.time_array) for p in particles]
        global_best = best_particles[np.argmin(best_fitnesses)]

        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()

                diff_personal = np.logical_xor(best_particles[i], particles[i]).astype(int)
                diff_global = np.logical_xor(global_best, particles[i]).astype(int)

                velocities[i] = 0.5 * velocities[i] + r1 * diff_personal + r2 * diff_global
                move = velocities[i] > 0
                particles[i] = np.logical_xor(particles[i], move)

                current_fitness = fitness(particles[i], self.coverage_matrix, self.time_array)
                if current_fitness < best_fitnesses[i]:
                    best_particles[i] = particles[i].copy()
                    best_fitnesses[i] = current_fitness

            global_best = best_particles[np.argmin(best_fitnesses)]

        return global_best