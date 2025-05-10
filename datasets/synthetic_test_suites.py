import pandas as pd
import random
import os

# --------- CONFIG ---------
SIZES = {
    "small": {"cases": 1000, "modules": 40},
    "medium": {"cases": 5000, "modules": 80},
    "large": {"cases": 10000, "modules": 120}
}
MODULES_PER_TEST = 1  # Still covering only one module per test for now

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
os.makedirs("generated_datasets", exist_ok=True)

for label, config in SIZES.items():
    num_cases = config["cases"]
    num_modules = config["modules"]
    print(f"\n🚀 Generating {label} dataset with {num_cases} tests and {num_modules} modules...")

    df_final = generate_test_cases(num_cases=num_cases, num_modules=num_modules)

    output_path = f"generated_datasets/{label}_realistic_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Saved {label} dataset to {output_path} ({len(df_final)} rows)")

print("\n🎉 All datasets with scaled module counts generated successfully!")
