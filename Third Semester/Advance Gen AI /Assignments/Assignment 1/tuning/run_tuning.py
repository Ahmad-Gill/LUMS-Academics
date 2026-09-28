import csv
import re
import subprocess
from pathlib import Path


# --------------------------------------------------
# Settings
# --------------------------------------------------

TRAIN_PATH = "data/tinystories/data/train.bin"
VAL_PATH = "data/tinystories/data/validation.bin"

STEPS = 1500

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 16

LEARNING_RATE_MAX = 3e-4
LEARNING_RATE_MIN = 3e-5
WARMUP_STEPS = 200
COSINE_STEPS = 9999

BETA1 = 0.9
BETA2 = 0.95

WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

SEED = 42

EVAL_INTERVAL = 100
LOG_INTERVAL = 5
CHECKPOINT_INTERVAL = 1500
NUM_VALIDATION_BATCHES = 10

DEVICE = "mps"


# --------------------------------------------------
# Experiments
# --------------------------------------------------

experiments = [
    # Baseline
    {
        "name": "baseline",
    },

    # Maximum learning rate
    {
        "name": "lr_1e-4",
        "learning_rate_max": 1e-4,
    },
    {
        "name": "lr_5e-4",
        "learning_rate_max": 5e-4,
    },

    # Minimum learning rate
    {
        "name": "min_lr_1e-5",
        "learning_rate_min": 1e-5,
    },
    {
        "name": "min_lr_5e-5",
        "learning_rate_min": 5e-5,
    },

    # Warmup
    {
        "name": "warmup_100",
        "warmup_steps": 100,
    },
    {
        "name": "warmup_400",
        "warmup_steps": 400,
    },

    # Beta 2
    {
        "name": "beta2_0.90",
        "beta2": 0.90,
    },
    {
        "name": "beta2_0.999",
        "beta2": 0.999,
    },

    # Weight decay
    {
        "name": "weight_decay_0.01",
        "weight_decay": 0.01,
    },
    {
        "name": "weight_decay_0.20",
        "weight_decay": 0.20,
    },

    # Gradient clipping
    {
        "name": "grad_clip_0.5",
        "max_grad_norm": 0.5,
    },
    {
        "name": "grad_clip_2.0",
        "max_grad_norm": 2.0,
    },

    # Effective batch size
    {
        "name": "effective_batch_128",
        "batch_size": 16,
        "gradient_accumulation_steps": 8,
    },
    {
        "name": "effective_batch_64",
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
    },
]


# --------------------------------------------------
# Directories
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = ROOT / "checkpoints"

LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

RESULTS_FILE = ROOT / "results.csv"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_value(experiment, key, default):
    return experiment.get(key, default)


def build_command(experiment, checkpoint_path):
    batch_size = get_value(
        experiment,
        "batch_size",
        BATCH_SIZE,
    )

    gradient_accumulation_steps = get_value(
        experiment,
        "gradient_accumulation_steps",
        GRADIENT_ACCUMULATION_STEPS,
    )

    learning_rate_max = get_value(
        experiment,
        "learning_rate_max",
        LEARNING_RATE_MAX,
    )

    learning_rate_min = get_value(
        experiment,
        "learning_rate_min",
        LEARNING_RATE_MIN,
    )

    warmup_steps = get_value(
        experiment,
        "warmup_steps",
        WARMUP_STEPS,
    )

    beta2 = get_value(
        experiment,
        "beta2",
        BETA2,
    )

    weight_decay = get_value(
        experiment,
        "weight_decay",
        WEIGHT_DECAY,
    )

    max_grad_norm = get_value(
        experiment,
        "max_grad_norm",
        MAX_GRAD_NORM,
    )

    command = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "src.train",

        "--train-path",
        TRAIN_PATH,

        "--val-path",
        VAL_PATH,

        "--num-steps",
        str(STEPS),

        "--batch-size",
        str(batch_size),

        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),

        "--learning-rate-max",
        str(learning_rate_max),

        "--learning-rate-min",
        str(learning_rate_min),

        "--warmup-steps",
        str(warmup_steps),

        "--cosine-steps",
        str(COSINE_STEPS),

        "--beta1",
        str(BETA1),

        "--beta2",
        str(beta2),

        "--eps",
        "1e-8",

        "--weight-decay",
        str(weight_decay),

        "--max-grad-norm",
        str(max_grad_norm),

        "--eval-interval",
        str(EVAL_INTERVAL),

        "--log-interval",
        str(LOG_INTERVAL),

        "--checkpoint-interval",
        str(CHECKPOINT_INTERVAL),

        "--num-validation-batches",
        str(NUM_VALIDATION_BATCHES),

        "--checkpoint-path",
        str(checkpoint_path),

        "--device",
        DEVICE,

        "--seed",
        str(SEED),
    ]

    return command


def parse_results(output):
    lines = output.splitlines()

    step_lines = [
        line
        for line in lines
        if line.startswith("step=")
    ]

    if not step_lines:
        return {
            "final_train_loss": "",
            "final_val_loss": "",
            "final_lr": "",
            "final_grad_norm": "",
            "best_val_loss": "",
        }

    train_losses = []
    val_losses = []

    last = step_lines[-1]

    train_match = re.search(
        r"train_loss=([0-9.eE+-]+)",
        last,
    )

    val_match = re.search(
        r"val_loss=([0-9.eE+-]+)",
        last,
    )

    lr_match = re.search(
        r"lr=([0-9.eE+-]+)",
        last,
    )

    grad_match = re.search(
        r"grad_norm=([0-9.eE+-]+)",
        last,
    )

    for line in step_lines:
        match = re.search(
            r"val_loss=([0-9.eE+-]+)",
            line,
        )

        if match:
            val_losses.append(
                float(match.group(1))
            )

    if train_match:
        final_train_loss = float(
            train_match.group(1)
        )
    else:
        final_train_loss = ""

    if val_match:
        final_val_loss = float(
            val_match.group(1)
        )
    else:
        final_val_loss = ""

    if lr_match:
        final_lr = float(
            lr_match.group(1)
        )
    else:
        final_lr = ""

    if grad_match:
        final_grad_norm = float(
            grad_match.group(1)
        )
    else:
        final_grad_norm = ""

    if val_losses:
        best_val_loss = min(val_losses)
    else:
        best_val_loss = ""

    return {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_lr": final_lr,
        "final_grad_norm": final_grad_norm,
        "best_val_loss": best_val_loss,
    }


# --------------------------------------------------
# CSV setup
# --------------------------------------------------

fieldnames = [
    "experiment",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "tokens_per_update",
    "total_sampled_tokens",
    "learning_rate_max",
    "learning_rate_min",
    "warmup_steps",
    "cosine_steps",
    "beta1",
    "beta2",
    "weight_decay",
    "max_grad_norm",
    "steps",
    "final_train_loss",
    "final_val_loss",
    "best_val_loss",
    "final_lr",
    "final_grad_norm",
    "status",
]


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

with RESULTS_FILE.open(
    "w",
    newline="",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):
        name = experiment["name"]

        print()
        print("=" * 70)
        print(
            f"Experiment {index}/{len(experiments)}: {name}"
        )
        print("=" * 70)

        batch_size = get_value(
            experiment,
            "batch_size",
            BATCH_SIZE,
        )

        gradient_accumulation_steps = get_value(
            experiment,
            "gradient_accumulation_steps",
            GRADIENT_ACCUMULATION_STEPS,
        )

        effective_batch_size = (
            batch_size
            * gradient_accumulation_steps
        )

        tokens_per_update = (
            effective_batch_size
            * 256
        )

        total_sampled_tokens = (
            tokens_per_update
            * STEPS
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"{name}.pt"
        )

        log_path = (
            LOG_DIR
            / f"{name}.log"
        )

        command = build_command(
            experiment,
            checkpoint_path,
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        print(
            f"Effective batch: {effective_batch_size}"
        )

        print(
            f"Tokens/update: {tokens_per_update:,}"
        )

        print(
            f"Total sampled tokens: "
            f"{total_sampled_tokens:,}"
        )

        print()

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines = []

            for line in process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)

            process.wait()

            output = "".join(output_lines)

            log_path.write_text(output)

            parsed = parse_results(output)

            status = (
                "success"
                if process.returncode == 0
                else "failed"
            )

        except Exception as error:
            output = str(error)

            log_path.write_text(output)

            parsed = {
                "final_train_loss": "",
                "final_val_loss": "",
                "best_val_loss": "",
                "final_lr": "",
                "final_grad_norm": "",
            }

            status = "failed"
        row = {
            "experiment": name,
            "batch_size": batch_size,
            "gradient_accumulation_steps":
                gradient_accumulation_steps,
            "effective_batch_size":
                effective_batch_size,
            "tokens_per_update":
                tokens_per_update,
            "total_sampled_tokens":
                total_sampled_tokens,
            "learning_rate_max":
                get_value(
                    experiment,
                    "learning_rate_max",
                    LEARNING_RATE_MAX,
                ),
            "learning_rate_min":
                get_value(
                    experiment,
                    "learning_rate_min",
                    LEARNING_RATE_MIN,
                ),
            "warmup_steps":
                get_value(
                    experiment,
                    "warmup_steps",
                    WARMUP_STEPS,
                ),
            "cosine_steps": COSINE_STEPS,
            "beta1": BETA1,
            "beta2":
                get_value(
                    experiment,
                    "beta2",
                    BETA2,
                ),
            "weight_decay":
                get_value(
                    experiment,
                    "weight_decay",
                    WEIGHT_DECAY,
                ),
            "max_grad_norm":
                get_value(
                    experiment,
                    "max_grad_norm",
                    MAX_GRAD_NORM,
                ),
            "steps": STEPS,
            **parsed,
            "status": status,
        }

        writer.writerow(row)
        csv_file.flush()

        print(
            f"Status: {status}"
        )

        print(
            f"Final train loss: "
            f"{parsed['final_train_loss']}"
        )

        print(
            f"Final validation loss: "
            f"{parsed['final_val_loss']}"
        )

        print(
            f"Best validation loss: "
            f"{parsed['best_val_loss']}"
        )

        print(
            f"Log: {log_path}"
        )

print()
print("=" * 70)
print("All tuning experiments completed.")
print("=" * 70)
print(f"Results: {RESULTS_FILE}")
print(f"Logs: {LOG_DIR}")
print(f"Checkpoints: {CHECKPOINT_DIR}")