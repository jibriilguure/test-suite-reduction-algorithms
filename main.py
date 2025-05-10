import os
import numpy as np
import time
import pandas as pd

from algorithms.aco import ACO_Optimizer
from algorithms.ga import GA_Optimizer
from algorithms.pso import PSO_Optimizer
from utils import fitness, is_full_coverage, load_dataset

# --- Main Execution ---

def print_results(name, solution, coverage_matrix, time_array, total_tests, elapsed_time):
    selected = np.sum(solution)
    reduction_ratio = (1 - selected / total_tests) * 100
    covered_modules = set()
    for i, selected_test in enumerate(solution):
        if selected_test:
            covered_modules.update(j for j, covers in enumerate(coverage_matrix[i]) if covers)
    print(f"\n{name} Results:")
    print(f"- Selected Test Cases: {selected} / {total_tests}")
    print(f"- Reduction Ratio: {reduction_ratio:.2f}%")
    print(f"- Coverage: {len(covered_modules)} / {coverage_matrix.shape[1]}")
    print(f"- Time Elapsed: {elapsed_time:.2f} seconds")

def save_selected_cases(name, solution, full_df):
    selected_df = full_df[solution].copy().reset_index(drop=True)
    output_path = f"results/selected_cases_{name.lower()}.csv"
    selected_df.to_csv(output_path, index=False)
    print(f"📁 Saved selected cases for {name} to {output_path}")

def main():
    dataset_path = "datasets/generated_datasets/small_realistic_dataset.csv"
    df_full = pd.read_csv(dataset_path)
    coverage_matrix, time_array = load_dataset(dataset_path)
    total_tests = coverage_matrix.shape[0]

    results = []  # To store results for all algorithms

    # Run GA
    ga = GA_Optimizer(coverage_matrix, time_array)
    start = time.time()
    ga_result = ga.run()
    end = time.time()
    ga_time = end - start
    print_results("GA", ga_result, coverage_matrix, time_array, total_tests, ga_time)
    results.append(("GA", np.sum(ga_result), total_tests, (1 - np.sum(ga_result) / total_tests) * 100,
                    len(set(j for i, s in enumerate(ga_result) if s for j, c in enumerate(coverage_matrix[i]) if c)),
                    ga_time))
    save_selected_cases("GA", ga_result, df_full)

    # Run PSO
    pso = PSO_Optimizer(coverage_matrix, time_array)
    start = time.time()
    pso_result = pso.run()
    end = time.time()
    pso_time = end - start
    print_results("PSO", pso_result, coverage_matrix, time_array, total_tests, pso_time)
    results.append(("PSO", np.sum(pso_result), total_tests, (1 - np.sum(pso_result) / total_tests) * 100,
                    len(set(j for i, s in enumerate(pso_result) if s for j, c in enumerate(coverage_matrix[i]) if c)),
                    pso_time))
    save_selected_cases("PSO", pso_result, df_full)

    # Run ACO
    aco = ACO_Optimizer(coverage_matrix, time_array)
    start = time.time()
    aco_result = aco.run()
    end = time.time()
    aco_time = end - start
    print_results("ACO", aco_result, coverage_matrix, time_array, total_tests, aco_time)
    results.append(("ACO", np.sum(aco_result), total_tests, (1 - np.sum(aco_result) / total_tests) * 100,
                    len(set(j for i, s in enumerate(aco_result) if s for j, c in enumerate(coverage_matrix[i]) if c)),
                    aco_time))
    save_selected_cases("ACO", aco_result, df_full)

    # Save summary to CSV
    df = pd.DataFrame(results, columns=["Algorithm", "Selected", "Total", "ReductionRatio", "Coverage", "TimeElapsed"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/experiment_log.csv", index=False)
    print("\n✅ Results saved to results/experiment_log.csv")

if __name__ == "__main__":
    main()