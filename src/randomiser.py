import argparse
import json
from pathlib import Path

import numpy as np

from data import STRUCTURE, parse_structure

def parse_args():
    """Reads command line arguments and returns the model output settings."""
    parser = argparse.ArgumentParser(description="Create a new random neural network model file.")
    parser.add_argument("model", help="Filename to write the new random model to.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing model file after confirmation.",
    )
    parser.add_argument(
        "--structure",
        default=",".join(map(str, STRUCTURE)),
        help="Comma-separated layer sizes. MNIST models must start with 784 and end with 10.",
    )
    parser.add_argument("--seed", type=int, help="Optional NumPy random seed for reproducible model weights.")
    parser.add_argument(
        "--initialisation",
        choices=("xavier", "normal"),
        default="xavier",
        help="Weight initialisation method. Defaults to xavier.",
    )
    return parser.parse_args()

def randomise_model(structure, initialisation="xavier"):
    """Creates random weights and biases for the given network structure."""
    if initialisation == "xavier":
        weights = [
            (np.random.randn(output_size, input_size) * np.sqrt(2 / (input_size + output_size))).tolist()
            for input_size, output_size in zip(structure[:-1], structure[1:])
        ]
    else:
        weights = [np.random.randn(output_size, input_size).tolist() for input_size, output_size in zip(structure[:-1], structure[1:])]

    return {
        "weights": weights,
        # Biases are stored as column vectors to match the matrix-based network shape.
        "biases": [np.zeros((size, 1)).tolist() for size in structure[1:]],
    }

def confirm_overwrite(model_path):
    """Asks the user to confirm before overwriting an existing model file."""
    response = input(f"Warning: '{model_path}' already exists and will be overwritten. Continue? (y/n): ")
    return response.strip().lower() == "y"

def main():
    """Creates a random model file from command line settings."""
    args = parse_args()
    model_path = Path(args.model)

    try:
        structure = parse_structure(args.structure)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if args.seed is not None:
        np.random.seed(args.seed)

    if model_path.exists():
        if not args.force:
            raise SystemExit(f"Error: '{model_path}' already exists. Use --force to overwrite it.")

        if not confirm_overwrite(model_path):
            raise SystemExit("Cancelled. No model file was changed.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "w") as file:
        json.dump(randomise_model(structure, args.initialisation), file, indent=4)

    print(f"Created random model: {model_path}")

if __name__ == "__main__":
    main()
