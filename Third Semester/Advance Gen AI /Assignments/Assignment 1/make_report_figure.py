import csv
from pathlib import Path

import matplotlib.pyplot as plt


results_path = Path("tuning/results.csv")
output_path = Path("report_assets/tuning_validation_loss.png")

output_path.parent.mkdir(parents=True, exist_ok=True)

experiments = []
validation_losses = []

with results_path.open() as f:
    reader = csv.DictReader(f)

    for row in reader:
        experiments.append(row["experiment"])
        validation_losses.append(float(row["final_val_loss"]))

# Sort from best (lowest loss) to worst.
pairs = sorted(zip(validation_losses, experiments))

validation_losses = [loss for loss, _ in pairs]
experiments = [name for _, name in pairs]

plt.figure(figsize=(10, 7))
plt.barh(experiments, validation_losses)
plt.xlabel("Final validation cross-entropy")
plt.ylabel("Experiment")
plt.title("Controlled hyperparameter experiments")
plt.tight_layout()
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Saved: {output_path}")