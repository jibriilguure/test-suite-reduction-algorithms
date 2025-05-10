import random

import numpy as np

from utils import fitness


class GA_Optimizer:
    def __init__(self, coverage_matrix, time_array, pop_size=50, generations=100):
        self.coverage_matrix = coverage_matrix
        self.time_array = time_array
        self.pop_size = pop_size
        self.generations = generations

    def run(self):
        population = [np.random.choice([False, True], size=len(self.coverage_matrix)) for _ in range(self.pop_size)]
        best_solution = None
        best_fitness = float('inf')

        for generation in range(self.generations):
            fitnesses = [fitness(ind, self.coverage_matrix, self.time_array) for ind in population]
            sorted_indices = np.argsort(fitnesses)
            population = [population[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]

            if fitnesses[0] < best_fitness:
                best_fitness = fitnesses[0]
                best_solution = population[0]

            next_population = population[:10]

            while len(next_population) < self.pop_size:
                parent1, parent2 = random.sample(population[:20], 2)
                child = np.array([random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)])
                if random.random() < 0.1:
                    idx = random.randint(0, len(child)-1)
                    child[idx] = not child[idx]
                next_population.append(child)

            population = next_population

        return best_solution
