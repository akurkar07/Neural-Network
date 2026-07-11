import argparse
import json
from pathlib import Path

import numpy as np

from data import STRUCTURE

def parse_args():
    """Reads command line arguments and returns the model output settings."""
    parser = argparse.ArgumentParser(description="Create a new random neural network model file.")
    parser.add_argument("model", help="Filename to write the new random model to.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing model file after confirmation.",
    )
    return parser.parse_args()

def randomise_model(structure):
    """Creates random weights and biases for the given network structure."""
    return {
        "weights": [np.random.randn(y, x).tolist() for x, y in zip(structure[:-1], structure[1:])],
        # Biases are stored as column vectors to match the matrix-based network shape.
        "biases": [np.random.randn(y, 1).tolist() for y in structure[1:]],
    }

def confirm_overwrite(model_path):
    """Asks the user to confirm before overwriting an existing model file."""
    response = input(f"Warning: '{model_path}' already exists and will be overwritten. Continue? (y/n): ")
    return response.strip().lower() == "y"

def main():
    """Creates a random model file from command line settings."""
    args = parse_args()
    model_path = Path(args.model)

    if model_path.exists():
        if not args.force:
            raise SystemExit(f"Error: '{model_path}' already exists. Use --force to overwrite it.")

        if not confirm_overwrite(model_path):
            raise SystemExit("Cancelled. No model file was changed.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "w") as file:
        json.dump(randomise_model(STRUCTURE), file, indent=4)

    print(f"Created random model: {model_path}")

if __name__ == "__main__":
    main()
