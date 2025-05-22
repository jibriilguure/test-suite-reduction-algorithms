import numpy as np
import pandas as pd

def fitness(solution, coverage_matrix, time_array, missing_penalty=1000):
    total_time = sum(time_array[i] for i, selected in enumerate(solution) if selected)
    if not is_full_coverage(solution, coverage_matrix):
        uncovered = coverage_matrix.shape[1] - np.sum(coverage_matrix[solution].any(axis=0))
        return float('inf') if uncovered == coverage_matrix.shape[1] else total_time + uncovered * missing_penalty
    return total_time



# slower
# def is_full_coverage(solution, coverage_matrix):
#     covered_modules = set()
#     for i, selected in enumerate(solution):
#         if selected:
#             covered_modules.update(j for j, covers in enumerate(coverage_matrix[i]) if covers)
#     return len(covered_modules) == coverage_matrix.shape[1]

#this fn is faster 
def is_full_coverage(sol: np.ndarray, cov_bool: np.ndarray) -> bool:
    return cov_bool[sol].any(axis=0).all()

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    module_cols = [col for col in df.columns if col.startswith('T-module-')]
    coverage_matrix = df[module_cols].values
    time_array = df['time_to_execute'].values
    return coverage_matrix, time_array
