import argparse
import os
import sys
from pathlib import Path

from benchmark import run_batch_benchmark
from data import STRUCTURE, load_mnist_data, load_model, save_model, validate_model_data
from dependencies import Network
from training import evaluate_network, save_metric_plot, train_network


def parse_args():
    """Reads command line arguments and returns the selected run settings."""
    parser = argparse.ArgumentParser(description="Train, test, or benchmark the neural network.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the neural network.")
    mode_group.add_argument("--test", action="store_true", help="Test the neural network.")
    mode_group.add_argument(
        "--benchmark-batches",
        action="store_true",
        help="Train fresh copies of the model with different batch sizes and compare stats.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print model details and each test prediction/cost.")
    parser.add_argument("--show-tf-logs", action="store_true", help="Show TensorFlow startup logs.")
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="Array backend for --train or --test. Defaults to numpy.",
    )
    parser.add_argument(
        "--backends",
        default="numpy",
        help="Comma-separated backends for --benchmark-batches, for example numpy,cupy.",
    )
    parser.add_argument("--model", default="data.json", help="Model file to load and save. Defaults to data.json.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs. Defaults to 30.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Training learning rate. Defaults to 0.1.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size. Defaults to 1, matching the original stochastic update behaviour.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,8,16,32,64,128,256",
        help="Comma-separated batch sizes to compare with --benchmark-batches.",
    )
    parser.add_argument("--benchmark-train-limit", type=int, help="Limit benchmark training to the first N examples.")
    parser.add_argument("--benchmark-test-limit", type=int, help="Limit benchmark evaluation to the first N examples.")
    parser.add_argument(
        "--benchmark-output",
        default="outputs/batch_benchmark_summary.csv",
        help="CSV path for benchmark summary stats.",
    )
    parser.add_argument(
        "--benchmark-history-output",
        default="outputs/batch_benchmark_history.csv",
        help="CSV path for per-epoch benchmark stats.",
    )
    parser.add_argument(
        "--benchmark-plot",
        default="outputs/batch_benchmark_stats.png",
        help="Image path for benchmark comparison graphs.",
    )
    parser.add_argument("--save-plot", help="Save an average training cost per epoch plot to the given image path.")
    parser.add_argument("--save-accuracy-plot", help="Save a test accuracy per epoch plot to the given image path.")

    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    """Validates parsed command line settings."""
    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0.")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0.")

    try:
        args.batch_sizes = [int(size.strip()) for size in args.batch_sizes.split(",") if size.strip()]
    except ValueError:
        parser.error("--batch-sizes must be a comma-separated list of integers.")

    if not args.batch_sizes or any(size <= 0 for size in args.batch_sizes):
        parser.error("--batch-sizes must contain at least one positive integer.")

    args.backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    valid_backends = {"numpy", "cupy"}
    if not args.backends or any(backend not in valid_backends for backend in args.backends):
        parser.error("--backends must contain one or more of: numpy, cupy.")

    if args.benchmark_train_limit is not None and args.benchmark_train_limit <= 0:
        parser.error("--benchmark-train-limit must be greater than 0.")

    if args.benchmark_test_limit is not None and args.benchmark_test_limit <= 0:
        parser.error("--benchmark-test-limit must be greater than 0.")


def train(args, data, network, train_inputs, train_outputs, test_inputs, test_outputs, test_y):
    """Runs training and saves the updated model."""
    print("Training")
    stats = train_network(
        network,
        train_inputs,
        train_outputs,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        progress=True,
        test_inputs=test_inputs if args.save_accuracy_plot else None,
        test_outputs=test_outputs if args.save_accuracy_plot else None,
        test_y=test_y if args.save_accuracy_plot else None,
    )

    print(f"Lowest Cost: {stats['lowest_cost']}, lowest Cost / example: {stats['lowest_cost'] / len(train_inputs)}")
    print(f"Training time: {stats['total_seconds']:.2f}s, Updates: {stats['updates']}, Batch size: {args.batch_size}")

    if args.save_plot:
        save_metric_plot(stats["average_costs"], args.save_plot, "Average cost per training example")
        print(f"Saved cost plot: {args.save_plot}")

    if args.save_accuracy_plot:
        save_metric_plot(stats["accuracies"], args.save_accuracy_plot, "Test accuracy (%)")
        print(f"Saved accuracy plot: {args.save_accuracy_plot}")

    save_model(Path(args.model), data, network)


def test(args, network, test_inputs, test_outputs, test_y):
    """Runs model evaluation."""
    print("Testing")
    no_of_examples = len(test_inputs)
    accuracy, wrong, average_cost = evaluate_network(network, test_inputs, test_outputs, test_y, args.verbose)
    correct = no_of_examples - wrong
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{no_of_examples})")
    print(f"Incorrect: {wrong}")
    print(f"Average cost/example: {average_cost}")


def main():
    """CLI entry point."""
    args = parse_args()

    if not args.show_tf_logs:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    model_path = Path(args.model)
    data = load_model(model_path)
    train_inputs, train_outputs, test_inputs, test_outputs, test_y = load_mnist_data()

    valid = validate_model_data(data, STRUCTURE)
    if args.verbose:
        print(f"Valid: {valid}")

    if not valid:
        sys.exit(
            f"Error: {model_path} weights/biases do not match the expected network "
            "structure. Run src/randomiser.py to regenerate the model."
        )

    network = Network(data, STRUCTURE, args.backend)
    if args.verbose:
        print(network)

    if args.train:
        train(args, data, network, train_inputs, train_outputs, test_inputs, test_outputs, test_y)
    elif args.benchmark_batches:
        run_batch_benchmark(args, data, STRUCTURE, train_inputs, train_outputs, test_inputs, test_outputs, test_y)
    else:
        test(args, network, test_inputs, test_outputs, test_y)


if __name__ == "__main__":
    main()
