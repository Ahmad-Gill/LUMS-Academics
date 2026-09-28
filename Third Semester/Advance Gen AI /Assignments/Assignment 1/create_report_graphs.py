import csv
from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

OUTPUT_DIR = Path("report_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Hyperparameter tuning validation loss
# --------------------------------------------------

results_path = Path("tuning/results.csv")

experiments = []
validation_losses = []

with results_path.open() as f:
    reader = csv.DictReader(f)

    for row in reader:
        experiments.append(row["experiment"])
        validation_losses.append(float(row["final_val_loss"]))

pairs = sorted(zip(validation_losses, experiments))

validation_losses = [loss for loss, _ in pairs]
experiments = [name for _, name in pairs]

plt.figure(figsize=(10, 7))
plt.barh(experiments, validation_losses)

plt.xlabel("Final validation cross-entropy")
plt.ylabel("Experiment")
plt.title("Hyperparameter Tuning Results")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "tuning_validation_loss.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# --------------------------------------------------
# 2. Final training loss
# --------------------------------------------------

steps = [
    9940,
    9950,
    9960,
    9970,
    9980,
    9990,
    10000,
]

training_losses = [
    1.6173,
    1.5965,
    1.6055,
    1.5590,
    1.6010,
    1.5829,
    1.5941,
]

plt.figure(figsize=(9, 5))

plt.plot(
    steps,
    training_losses,
    marker="o",
)

plt.xlabel("Optimizer step")
plt.ylabel("Training loss")
plt.title("Training Loss Near the End of Final Run")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_training_loss.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 3. Learning-rate schedule
# --------------------------------------------------

total_steps = 10000
warmup_steps = 100
max_lr = 5e-4
min_lr = 3e-5

schedule_steps = []
learning_rates = []

for step in range(total_steps + 1):

    if step <= warmup_steps:
        lr = max_lr * step / warmup_steps

    else:
        progress = (step - warmup_steps) / (
            total_steps - warmup_steps
        )

        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + __import__("math").cos(__import__("math").pi * progress)
        )

    schedule_steps.append(step)
    learning_rates.append(lr)

plt.figure(figsize=(9, 5))

plt.plot(
    schedule_steps,
    learning_rates,
)

plt.xlabel("Optimizer step")
plt.ylabel("Learning rate")
plt.title("Learning Rate Schedule")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "learning_rate_schedule.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Done
# --------------------------------------------------

print("Created:")
print(" - report_assets/tuning_validation_loss.png")
print(" - report_assets/final_training_loss.png")
print(" - report_assets/learning_rate_schedule.png")