import pandas as pd
import random
import os

# --------- CONFIG ---------
DATASET_CONFIGS = {
    "small":  {"total": 1000, "redundancy": 0.10, "modules": 2000},
    "medium": {"total": 3000, "redundancy": 0.20, "modules": 4800},
    "large":  {"total": 5000, "redundancy": 0.30, "modules": 8000}
}

# Adjustable module coverage range
MODULES_PER_TEST_RANGE = (1 , 3)

# --------- Helper Functions ---------
def format_module_name(index):
    return f"T-module-{index+1:04d}"

def generate_unique_case(module_names):
    row = {mod: 0 for mod in module_names}
    num_covered = random.randint(*MODULES_PER_TEST_RANGE)
    selected = random.sample(module_names, num_covered)
    for mod in selected:
        row[mod] = 1
    row["time_to_execute"] = round(random.uniform(0.2, 5.0), 2)
    row["priority"] = random.choice(["High", "Medium", "Low"])
    return row

def generate_dataset(total, redundancy, num_modules):
    num_unique = int(total * (1 - redundancy))
    num_duplicates = total - num_unique

    module_names = [format_module_name(i) for i in range(num_modules)]
    unique_cases = [generate_unique_case(module_names) for _ in range(num_unique)]

    # Ensure all modules are covered at least once
    covered_modules = set()
    for case in unique_cases:
        covered_modules.update(
            int(col.split("-")[-1]) - 1
            for col, val in case.items() if col.startswith("T-module-") and val == 1
        )

    all_modules = set(range(num_modules))
    missing_modules = list(all_modules - covered_modules)

    for m in missing_modules:
        patch = {mod: 0 for mod in module_names}
        patch[module_names[m]] = 1
        patch["time_to_execute"] = round(random.uniform(0.2, 5.0), 2)
        patch["priority"] = random.choice(["High", "Medium", "Low"])
        unique_cases.append(patch)

    # Final data construction
    data = unique_cases.copy()

    # Add duplicates
    for _ in range(num_duplicates):
        original = random.choice(unique_cases).copy()
        original["time_to_execute"] = round(original["time_to_execute"] + random.uniform(-0.1, 0.1), 2)
        original["priority"] = random.choice(["High", "Medium", "Low"])
        data.append(original)

    # Build DataFrame
    df = pd.DataFrame(data)
    df.insert(0, "test_id", [f"TC_{i+1:05d}" for i in range(len(data))])

    # Reorder columns
    fixed_cols = ["test_id", "time_to_execute", "priority"]
    module_cols = [col for col in df.columns if col.startswith("T-module-")]
    df = df[fixed_cols + module_cols]

    return df

# --------- MAIN ---------
output_dir = "generated_datasets"
os.makedirs(output_dir, exist_ok=True)

generated_files = []

for label, config in DATASET_CONFIGS.items():
    df = generate_dataset(config["total"], config["redundancy"], config["modules"])
    file_path = f"{output_dir}/{label}_realistic_dataset.csv"
    df.to_csv(file_path, index=False)
    generated_files.append((label, file_path, df.shape))
    print(f"✅ {label.capitalize()} dataset generated at: {file_path} ({df.shape[0]} rows, {df.shape[1] - 3} modules)")
