import numpy as np
from utils import fitness

class ACO_Optimizer:
    def __init__(self, coverage_matrix, time_array,
                 num_ants=5,
                 num_iterations=50,
                 alpha=1.0,
                 beta=2.0,
                 evaporation_rate=0.3,
                 patience=5
                 ):
        self.coverage_matrix = coverage_matrix
        self.time_array = time_array
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones(coverage_matrix.shape[0])
        self.best_solution = None
        self.best_fitness = float('inf')
        self.patience = patience  # Early stopping patience

    def run(self):
        no_improvement = 0

        for iteration in range(self.num_iterations):
            all_solutions = []
            all_fitnesses = []
            improvement = False

            for _ in range(self.num_ants):
                solution = np.zeros(len(self.coverage_matrix), dtype=bool)
                covered_modules = set()
                max_steps = len(self.coverage_matrix)  # safeguard

                for _ in range(max_steps):
                    if len(covered_modules) >= self.coverage_matrix.shape[1]:
                        break

                    probabilities = []
                    for i in range(len(self.coverage_matrix)):
                        if not solution[i]:
                            uncovered = sum(
                                1 for m in range(self.coverage_matrix.shape[1])
                                if self.coverage_matrix[i, m] and m not in covered_modules
                            )
                            heuristic = (uncovered + 1e-6) / (self.time_array[i] + 0.1)
                            probabilities.append((self.pheromone[i] ** self.alpha) * (heuristic ** self.beta))
                        else:
                            probabilities.append(0)

                    probabilities = np.array(probabilities)
                    if probabilities.sum() == 0:
                        break
                    probabilities /= probabilities.sum()
                    selected = np.random.choice(len(self.coverage_matrix), p=probabilities)

                    solution[selected] = True
                    covered_modules.update(
                        m for m in range(self.coverage_matrix.shape[1])
                        if self.coverage_matrix[selected, m]
                    )

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

            if improvement:
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.patience:
                print(f"⏹️ vv  Early stopping at iteration {iteration+1}")
                break

        return self.best_solution
