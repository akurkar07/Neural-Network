import copy
import csv
import statistics
from pathlib import Path

from dependencies import Network, get_backend
from training import evaluate_network, train_network, warm_up_network


def save_benchmark_records(summary_records, history_records, summary_path, history_path):
    """Writes benchmark summary and per-epoch records as CSV files."""
    summary_path = Path(summary_path)
    history_path = Path(history_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_records[0].keys())
        writer.writeheader()
        writer.writerows(summary_records)

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history_records[0].keys())
        writer.writeheader()
        writer.writerows(history_records)


def add_summary_statistics(summary_records):
    """Adds median and range values for each backend and batch-size configuration."""
    configurations = {(record["backend"], record["batch_size"]) for record in summary_records}
    for backend, batch_size in configurations:
        records = [
            record
            for record in summary_records
            if record["backend"] == backend and record["batch_size"] == batch_size
        ]
        durations = [record["total_seconds"] for record in records]
        throughputs = [record["examples_per_second"] for record in records]
        for record in records:
            record["median_total_seconds"] = statistics.median(durations)
            record["minimum_total_seconds"] = min(durations)
            record["maximum_total_seconds"] = max(durations)
            record["median_examples_per_second"] = statistics.median(throughputs)


def save_benchmark_plot(summary_records, history_records, plot_path):
    """Saves comparison graphs for benchmark records."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    configurations = sorted({(record["backend"], record["batch_size"], record["run"]) for record in history_records})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Training backend and batch size benchmark")

    for backend, batch_size, run in configurations:
        records = [
            record
            for record in history_records
            if record["backend"] == backend and record["batch_size"] == batch_size and record["run"] == run
        ]
        epochs = [record["epoch"] for record in records]
        label = f"{backend}, batch {batch_size}, run {run}"

        axes[0, 0].plot(epochs, [record["train_average_cost"] for record in records], marker="o", label=label)
        axes[0, 0].plot(
            epochs,
            [record["test_average_cost"] for record in records],
            marker="x",
            linestyle="--",
            label=f"{label} test",
        )
        axes[0, 1].plot(epochs, [record["test_accuracy"] for record in records], marker="o", label=label)
        axes[1, 0].plot(epochs, [record["epoch_seconds"] for record in records], marker="o", label=label)

    axes[0, 0].set_title("Train/test cost")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Average cost/example")
    axes[0, 0].legend()

    axes[0, 1].set_title("Test accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()

    axes[1, 0].set_title("Time per epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].legend()

    labels = [f"{record['backend']}\nbatch {record['batch_size']}" for record in summary_records]
    throughput = [record["examples_per_second"] for record in summary_records]
    speedups = [
        record["examples_per_second"] / summary_records[0]["examples_per_second"]
        for record in summary_records
    ]
    bars = axes[1, 1].bar(labels, throughput)
    axes[1, 1].set_title("Throughput and speedup")
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylabel("Examples/second")
    for bar, speedup in zip(bars, speedups):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
        )

    updates_per_epoch = [record["updates"] / record["epochs"] for record in summary_records]
    axes[0, 2].bar(labels, updates_per_epoch)
    axes[0, 2].set_title("Updates per epoch")
    axes[0, 2].set_xlabel("Batch size")
    axes[0, 2].set_ylabel("Weight updates")

    baseline = summary_records[0]
    best_accuracy = max(summary_records, key=lambda record: record["test_accuracy"])
    fastest = max(summary_records, key=lambda record: record["examples_per_second"])
    summary_lines = [
        f"Epochs: {baseline['epochs']}",
        f"Learning rate: {baseline['learning_rate']}",
        f"Model: {baseline['model']}",
        f"Train examples: {baseline['train_examples']}",
        f"Test examples: {baseline['test_examples']}",
        "",
        f"Fastest: {fastest['backend']} batch {fastest['batch_size']} ({speedups[summary_records.index(fastest)]:.1f}x)",
        f"Best accuracy: {best_accuracy['backend']} batch {best_accuracy['batch_size']} ({best_accuracy['test_accuracy']:.2f}%)",
        "",
        "Final rows:",
    ]
    for record in summary_records:
        summary_lines.append(
                f"{record['backend']} batch {record['batch_size']}: "
            f"acc {record['test_accuracy']:.2f}%, "
            f"wrong {record['wrong']}/{record['test_examples']}, "
            f"time {record['total_seconds']:.2f}s"
        )

    axes[1, 2].axis("off")
    axes[1, 2].set_title("Summary")
    axes[1, 2].text(
        0,
        0.95,
        "\n".join(summary_lines),
        transform=axes[1, 2].transAxes,
        va="top",
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def run_batch_benchmark(args, data, structure, train_inputs, train_outputs, test_inputs, test_outputs, test_y):
    """Benchmarks training with multiple batch sizes from the same starting model."""
    benchmark_train_inputs = train_inputs
    benchmark_train_outputs = train_outputs
    benchmark_test_inputs = test_inputs
    benchmark_test_outputs = test_outputs
    benchmark_test_y = test_y

    if args.benchmark_train_limit is not None:
        benchmark_train_inputs = benchmark_train_inputs[:args.benchmark_train_limit]
        benchmark_train_outputs = benchmark_train_outputs[:args.benchmark_train_limit]

    if args.benchmark_test_limit is not None:
        benchmark_test_inputs = benchmark_test_inputs[:args.benchmark_test_limit]
        benchmark_test_outputs = benchmark_test_outputs[:args.benchmark_test_limit]
        benchmark_test_y = benchmark_test_y[:args.benchmark_test_limit]

    print("Training backend and batch size benchmark")
    print(
        f"Epochs: {args.epochs}, LR: {args.learning_rate}, "
        f"Train examples: {len(benchmark_train_inputs)}, Test examples: {len(benchmark_test_inputs)}"
    )
    print(
        "backend,batch_size,run,updates,total_seconds,seconds_per_epoch,examples_per_second,"
        "train_average_cost,test_accuracy,test_average_cost,wrong"
    )
    summary_records = []
    history_records = []

    for backend in args.backends:
        xp = get_backend(backend)
        backend_train_inputs = xp.asarray(benchmark_train_inputs, dtype=xp.float32)
        backend_train_outputs = xp.asarray(benchmark_train_outputs, dtype=xp.float32)
        backend_test_inputs = xp.asarray(benchmark_test_inputs, dtype=xp.float32)
        backend_test_outputs = xp.asarray(benchmark_test_outputs, dtype=xp.float32)
        backend_test_y = xp.asarray(benchmark_test_y)

        for batch_size in args.batch_sizes:
            for run in range(1, args.benchmark_runs + 1):
                if backend == "cupy":
                    for _ in range(args.benchmark_warmup_batches):
                        warm_up_network(
                            Network(copy.deepcopy(data), structure, backend),
                            backend_train_inputs,
                            backend_train_outputs,
                            batch_size,
                        )

                network = Network(copy.deepcopy(data), structure, backend)
                stats = train_network(
                    network,
                    backend_train_inputs,
                    backend_train_outputs,
                    args.epochs,
                    args.learning_rate,
                    batch_size,
                    test_inputs=backend_test_inputs,
                    test_outputs=backend_test_outputs,
                    test_y=backend_test_y,
                )
                accuracy, wrong, test_average_cost = evaluate_network(
                    network,
                    backend_test_inputs,
                    backend_test_outputs,
                    backend_test_y,
                )
                examples_seen = len(benchmark_train_inputs) * args.epochs
                examples_per_second = examples_seen / stats["total_seconds"]
                seconds_per_epoch = stats["total_seconds"] / args.epochs
                train_average_cost = stats["average_costs"][-1]

                summary_record = {
                    "backend": backend,
                    "batch_size": batch_size,
                    "run": run,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "model": args.model,
                    "train_examples": len(benchmark_train_inputs),
                    "test_examples": len(benchmark_test_inputs),
                    "updates": stats["updates"],
                    "total_seconds": stats["total_seconds"],
                    "seconds_per_epoch": seconds_per_epoch,
                    "examples_per_second": examples_per_second,
                    "train_average_cost": train_average_cost,
                    "test_accuracy": accuracy,
                    "test_average_cost": float(test_average_cost),
                    "wrong": wrong,
                }
                summary_records.append(summary_record)

                for epoch, (train_cost, epoch_seconds, epoch_accuracy, epoch_test_cost, epoch_wrong) in enumerate(
                    zip(
                        stats["average_costs"],
                        stats["epoch_seconds"],
                        stats["accuracies"],
                        stats["test_average_costs"],
                        stats["wrong_counts"],
                    ),
                    start=1,
                ):
                    history_records.append(
                        {
                            "backend": backend,
                            "batch_size": batch_size,
                            "run": run,
                            "epoch": epoch,
                            "train_average_cost": train_cost,
                            "test_accuracy": epoch_accuracy,
                            "test_average_cost": epoch_test_cost,
                            "wrong": epoch_wrong,
                            "epoch_seconds": epoch_seconds,
                        }
                    )

                print(
                    f"{backend},"
                    f"{batch_size},"
                    f"{run},"
                    f"{stats['updates']},"
                    f"{stats['total_seconds']:.4f},"
                    f"{seconds_per_epoch:.4f},"
                    f"{examples_per_second:.2f},"
                    f"{train_average_cost:.6f},"
                    f"{accuracy:.2f},"
                    f"{test_average_cost:.6f},"
                    f"{wrong}"
                )

    add_summary_statistics(summary_records)
    save_benchmark_records(summary_records, history_records, args.benchmark_output, args.benchmark_history_output)
    save_benchmark_plot(summary_records, history_records, args.benchmark_plot)
    print(f"Saved benchmark summary: {args.benchmark_output}")
    print(f"Saved benchmark history: {args.benchmark_history_output}")
    print(f"Saved benchmark plot: {args.benchmark_plot}")
