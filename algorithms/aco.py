import numpy as np
import time
from utils import fitness

class ACO_Optimizer:
    def __init__(self, coverage_matrix, time_array,
                 alpha=1.0,
                 beta=2.0,
                 evaporation_rate=0.3,
                 patience=5,
                 max_runtime=600  # 10-minute cap for large datasets
                 ):
        self.coverage_matrix = coverage_matrix
        self.time_array = time_array
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.patience = patience
        self.max_runtime = max_runtime

        self.num_tests = coverage_matrix.shape[0]
        self.num_modules = coverage_matrix.shape[1]

        # Scale num_ants and iterations based on size
        if self.num_tests > 5000:
            self.num_ants = 3
            self.num_iterations = 30
        else:
            self.num_ants = 5
            self.num_iterations = 50

        self.pheromone = np.ones(self.num_tests)
        self.best_solution = None
        self.best_fitness = float('inf')

        # Cache module indices per test case
        self.module_indices = [set(np.where(row)[0]) for row in coverage_matrix]

    def run(self):
        no_improvement = 0
        start_time = time.time()

        for iteration in range(self.num_iterations):
            if time.time() - start_time > self.max_runtime:
                print(f"⏹️ Stopping early due to timeout at iteration {iteration + 1}")
                break

            all_solutions = []
            all_fitnesses = []
            improvement = False

            for _ in range(self.num_ants):
                solution = np.zeros(self.num_tests, dtype=bool)
                covered_modules = set()

                for _ in range(self.num_tests):
                    if len(covered_modules) >= self.num_modules:
                        break

                    uncovered_counts = np.array([
                        len(mods - covered_modules) if not solution[i] else 0
                        for i, mods in enumerate(self.module_indices)
                    ])

                    heuristics = (uncovered_counts + 1e-6) / (self.time_array + 0.1)
                    probabilities = np.where(
                        solution,
                        0,
                        (self.pheromone ** self.alpha) * (heuristics ** self.beta)
                    )

                    total = probabilities.sum()
                    if total == 0:
                        break
                    probabilities /= total

                    selected = np.random.choice(self.num_tests, p=probabilities)
                    solution[selected] = True
                    covered_modules.update(self.module_indices[selected])

                sol_fitness = fitness(solution, self.coverage_matrix, self.time_array)
                all_solutions.append(solution)
                all_fitnesses.append(sol_fitness)

                if sol_fitness < self.best_fitness:
                    self.best_fitness = sol_fitness
                    self.best_solution = solution.copy()
                    improvement = True

            # Update pheromones
            self.pheromone *= (1 - self.evaporation_rate)
            for sol, fit in zip(all_solutions, all_fitnesses):
                if fit < float('inf'):
                    self.pheromone[sol] += 1.0 / fit

            no_improvement = 0 if improvement else no_improvement + 1
            if no_improvement >= self.patience:
                print(f"⏹️ Early stopping due to no improvement at iteration {iteration + 1}")
                break

        return self.best_solution
