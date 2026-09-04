import json
from pathlib import Path

import numpy as np


STRUCTURE = [784, 16, 16, 10]


def parse_structure(value):
    """Parses a comma-separated network structure and validates MNIST endpoints."""
    try:
        structure = [int(size.strip()) for size in value.split(",") if size.strip()]
    except ValueError as error:
        raise ValueError("Structure must be a comma-separated list of positive integers.") from error

    if len(structure) < 2 or any(size <= 0 for size in structure):
        raise ValueError("Structure must contain at least two positive layer sizes.")
    if structure[0] != 784 or structure[-1] != 10:
        raise ValueError("MNIST structures must start with 784 inputs and end with 10 outputs.")
    return structure


def validate_model_data(data, structure=STRUCTURE):
    """Checks saved weights and biases match the expected network structure."""
    if "weights" not in data or "biases" not in data:
        return False

    expected_layers = list(zip(structure[:-1], structure[1:]))
    if len(data["weights"]) != len(expected_layers) or len(data["biases"]) != len(expected_layers):
        return False

    for count, (previous_layer_size, current_layer_size) in enumerate(expected_layers):
        weight_layer = data["weights"][count]
        if len(weight_layer) != current_layer_size:
            return False

        for weights in weight_layer:
            if len(weights) != previous_layer_size:
                return False

    for count, (_, current_layer_size) in enumerate(expected_layers):
        bias_layer = data["biases"][count]
        if len(bias_layer) != current_layer_size:
            return False

        for bias in bias_layer:
            if isinstance(bias, list) and len(bias) != 1:
                return False

    return True


def load_model(model_path):
    """Loads a saved JSON model file."""
    with open(model_path, "r") as file:
        return json.load(file)


def save_model(model_path, data, network):
    """Saves updated weights and biases back to a JSON model file."""
    if network.backend == "cupy":
        data["weights"] = [network.xp.asnumpy(weights).tolist() for weights in network.weights]
        data["biases"] = [network.xp.asnumpy(biases).tolist() for biases in network.biases]
    else:
        data["weights"] = [weights.tolist() for weights in network.weights]
        data["biases"] = [biases.tolist() for biases in network.biases]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w") as file:
        json.dump(data, file, indent=4)


def load_mnist_data():
    """Loads MNIST and returns flattened, normalised train/test arrays."""
    from tensorflow.keras.datasets import mnist

    (input_data, desired_outputs), (test_x, test_y) = mnist.load_data()
    max_range = 255
    train_inputs = input_data.reshape(len(input_data), -1).astype(np.float32) / max_range
    train_outputs = np.eye(10, dtype=np.float32)[desired_outputs]
    test_inputs = test_x.reshape(len(test_x), -1).astype(np.float32) / max_range
    test_outputs = np.eye(10, dtype=np.float32)[test_y]
    return train_inputs, train_outputs, test_inputs, test_outputs, test_y
