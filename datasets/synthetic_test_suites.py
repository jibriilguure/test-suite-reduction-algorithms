import pandas as pd
import random

# --------- CONFIG ---------
NUM_MODULES = 40
SIZES = {
    "small": 1000,
    "medium": 5000,
    "large": 10000
}
MODULES_PER_TEST = 1  # Only 1 module per test case

# --------- Generate Base Dataset ---------
def generate_test_cases(num_cases, num_modules):
    data = []

    for i in range(num_cases):
        row = {f"covers_module_{j+1}": 0 for j in range(num_modules)}
        module = random.randint(0, num_modules - 1)
        row[f"covers_module_{module + 1}"] = 1

        row["time_to_execute"] = round(random.uniform(0.2, 5.0), 2)
        row["priority"] = random.choice(["High", "Medium", "Low"])
        data.append(row)

    df = pd.DataFrame(data)
    df.insert(0, "test_id", [f"TC_{i:05d}" for i in range(1, len(df) + 1)])
    cols = ["test_id", "time_to_execute", "priority"] + [col for col in df.columns if col.startswith("covers_module_")]
    return df[cols].reset_index(drop=True)

# --------- MAIN ---------
for label, total_tests in SIZES.items():
    print(f"\n🚀 Generating {label} dataset with {total_tests} unique tests...")

    df_final = generate_test_cases(num_cases=total_tests, num_modules=NUM_MODULES)

    output_path = f"generated_datasets/{label}_realistic_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Saved {label} dataset to {output_path} ({len(df_final)} rows)")

print("\n🎉 All datasets with only one module per test generated successfully!")
