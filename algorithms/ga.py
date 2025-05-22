import random
import numpy as np
from utils import fitness


class GA_Optimizer:
    def __init__(self, coverage_matrix,
                 time_array,
                 pop_size=5,
                 generations=10, early_stop_patience=3):
        self.coverage_matrix = coverage_matrix
        self.time_array = time_array
        self.pop_size = pop_size
        self.generations = generations
        self.early_stop_patience = early_stop_patience
        self.num_modules = coverage_matrix.shape[1]
        self.num_tests = coverage_matrix.shape[0]

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

    def mutate(self, individual, mutation_rate=0.01):
        mask = np.random.rand(len(individual)) < mutation_rate
        individual[mask] = ~individual[mask]
        return individual

    def crossover(self, parent1, parent2):
        mask = np.random.rand(len(parent1)) > 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def run(self):
        # Initialize population
        population = [np.random.choice([False, True], size=self.num_tests) for _ in range(self.pop_size - 1)]
        population.insert(0, self.greedy_solution())  # Insert greedy solution

        best_solution = None
        best_fitness = float('inf')
        no_improvement = 0

        for generation in range(self.generations):
            fitnesses = [fitness(ind, self.coverage_matrix, self.time_array) for ind in population]
            sorted_indices = np.argsort(fitnesses)
            population = [population[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]

            if fitnesses[0] < best_fitness:
                best_fitness = fitnesses[0]
                best_solution = population[0].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.early_stop_patience:
                print(f"GA Early stopping at generation {generation + 1}")
                break

            next_population = [population[0]]  # Elitism: keep best

            while len(next_population) < self.pop_size:
                parent1, parent2 = random.sample(population[:20], 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                # Optional: repair solution to ensure full coverage
                # child = self.repair(child)

                next_population.append(child)

            population = next_population

        return best_solution
