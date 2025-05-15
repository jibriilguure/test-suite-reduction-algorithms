import pandas as pd
import random
import os

# --------- CONFIG ---------
SIZES = {
    "small": {"cases": 1000, "modules": 40},
    "medium": {"cases": 5000, "modules": 80},
    "large": {"cases": 10000, "modules": 120}
}

MODULES_PER_TEST_RANGE = (1, 40)  # Random 1 to 3 modules
REDUNDANCY_PERCENT = 0.1  # 10% of test cases will be redundant

# --------- Generate Dataset with Redundancy ---------
def generate_test_cases(num_cases, num_modules, redundancy_percent):
    unique_cases = int(num_cases * (1 - redundancy_percent))
    redundant_cases = num_cases - unique_cases
    data = []

    # Step 1: Generate unique test cases
    for _ in range(unique_cases):
        row = {f"covers_module_{j+1}": 0 for j in range(num_modules)}
        num_modules_covered = random.randint(*MODULES_PER_TEST_RANGE)
        selected_modules = random.sample(range(num_modules), num_modules_covered)
        for m in selected_modules:
            row[f"covers_module_{m+1}"] = 1
        row["time_to_execute"] = round(random.uniform(0.2, 5.0), 2)
        row["priority"] = random.choice(["High", "Medium", "Low"])
        data.append(row)

    # Step 2: Add redundant (copied) test cases from existing ones
    for _ in range(redundant_cases):
        original = random.choice(data).copy()
        # You can optionally vary time/priority to simulate slight variation
        original["time_to_execute"] = round(original["time_to_execute"] + random.uniform(-0.1, 0.1), 2)
        original["priority"] = random.choice(["High", "Medium", "Low"])
        data.append(original)

    df = pd.DataFrame(data)
    df.insert(0, "test_id", [f"TC_{i:05d}" for i in range(1, len(df) + 1)])
    cols = ["test_id", "time_to_execute", "priority"] + [col for col in df.columns if col.startswith("covers_module_")]
    return df[cols].reset_index(drop=True)

# --------- MAIN ---------
os.makedirs("generated_datasets", exist_ok=True)

for label, config in SIZES.items():
    num_cases = config["cases"]
    num_modules = config["modules"]
    print(f"\n🚀 Generating {label} dataset with redundancy...")

    df_final = generate_test_cases(num_cases, num_modules, REDUNDANCY_PERCENT)

    output_path = f"generated_datasets/{label}_realistic_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Saved {label} dataset to {output_path} ({len(df_final)} rows)")

print("\n🎉 All datasets with intentional redundancy generated successfully!")
