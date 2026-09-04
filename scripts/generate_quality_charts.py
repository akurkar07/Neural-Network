import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    """Reads the benchmark CSV paths and output directory."""
    parser = argparse.ArgumentParser(description="Create display charts from a quality benchmark.")
    parser.add_argument("history", help="Path to the per-epoch benchmark CSV.")
    parser.add_argument("summary", help="Path to the benchmark summary CSV.")
    parser.add_argument("output_directory", help="Directory for generated PNG files.")
    return parser.parse_args()


def load_records(path):
    """Loads CSV records from a benchmark output file."""
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def save_accuracy_chart(history, output_path):
    """Saves test accuracy by epoch for every tested batch size."""
    batches = sorted({int(row["batch_size"]) for row in history})
    colours = {32: "#2563eb", 64: "#f97316", 128: "#16a34a"}

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6))
    for batch_size in batches:
        records = [row for row in history if int(row["batch_size"]) == batch_size]
        axis.plot(
            [int(row["epoch"]) for row in records],
            [float(row["test_accuracy"]) for row in records],
            label=f"Batch {batch_size}",
            color=colours.get(batch_size),
            linewidth=2.5,
        )

    axis.axhline(92, color="#374151", linestyle="--", linewidth=1.5, label="92% target")
    axis.set(
        title="Medium model convergence on GPU",
        xlabel="Epoch",
        ylabel="Test accuracy (%)",
        xlim=(1, 30),
        ylim=(40, 97),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_trade_off_chart(summary, output_path):
    """Saves the final accuracy and training-time trade-off by batch size."""
    colours = {32: "#2563eb", 64: "#f97316", 128: "#16a34a"}
    records = sorted(summary, key=lambda row: int(row["batch_size"]))

    figure, axis = plt.subplots(figsize=(10, 6))
    for row in records:
        batch_size = int(row["batch_size"])
        training_time = float(row["total_seconds"])
        accuracy = float(row["test_accuracy"])
        axis.scatter(training_time, accuracy, s=180, color=colours.get(batch_size), zorder=3)
        axis.annotate(
            f"Batch {batch_size}\n{accuracy:.2f}%",
            (training_time, accuracy),
            xytext=(8, 8),
            textcoords="offset points",
        )

    axis.set(
        title="GPU training time and final accuracy",
        xlabel="Warmed training time for 30 epochs (seconds)",
        ylabel="Final test accuracy (%)",
        xlim=(0, 175),
        ylim=(91.5, 95.7),
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    """Creates the display charts from benchmark records."""
    args = parse_args()
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    save_accuracy_chart(load_records(args.history), output_directory / "gpu_medium_quality_accuracy.png")
    save_trade_off_chart(load_records(args.summary), output_directory / "gpu_medium_quality_tradeoff.png")


if __name__ == "__main__":
    main()