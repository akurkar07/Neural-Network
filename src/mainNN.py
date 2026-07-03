import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from NNDependencies import Network, cost

def parse_args():
    """Reads command line arguments and returns the selected run settings."""
    parser = argparse.ArgumentParser(description="Train or test the neural network.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the neural network.")
    mode_group.add_argument("--test", action="store_true", help="Test the neural network.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model details and each test prediction/cost.",
    )
    parser.add_argument(
        "--show-tf-logs",
        action="store_true",
        help="Show TensorFlow startup logs.",
    )
    parser.add_argument(
        "--model",
        default="data.json",
        help="Model file to load and save. Defaults to data.json.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. Defaults to 30.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Training learning rate. Defaults to 0.1.",
    )
    parser.add_argument(
        "--save-plot",
        help="Save a training cost plot to the given image path.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0.")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0.")

    return args

def save_cost_plot(costs, plot_path):
    """Saves a plot of average training cost per example over each epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(costs) + 1)
    plt.figure()
    plt.plot(epochs, costs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average cost per example")
    plt.title("Training cost over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def validate_model_data(data, structure):
    """Checks saved weights and biases match the expected network structure."""
    # These checks make sure the dimensions of the json's data are present and correct.
    if "weights" not in data or "biases" not in data:
        return False

    expected_layers = list(zip(structure[:-1], structure[1:]))

    # Checks no. of layers in JSON matches structure.
    if len(data["weights"]) != len(expected_layers) or len(data["biases"]) != len(expected_layers):
        return False

    # Checks each weight matrix has one row per current-layer neurone and one column per previous-layer neurone.
    for count, (previous_layer_size, current_layer_size) in enumerate(expected_layers):
        weight_layer = data["weights"][count]
        if len(weight_layer) != current_layer_size:
            return False

        for weights in weight_layer:
            if len(weights) != previous_layer_size:
                return False

    # Checks each bias layer is stored as a column vector: [[bias], [bias], ...].
    for count, (_, current_layer_size) in enumerate(expected_layers):
        bias_layer = data["biases"][count]
        if len(bias_layer) != current_layer_size:
            return False

        for bias in bias_layer:
            if not isinstance(bias, list) or len(bias) != 1:
                return False

    return True

args = parse_args()

if not args.show_tf_logs: # Hides the tensorflow logs if --show-tf-logs isn't enabled
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# TensorFlow is a big import so it is imported after CLI parsing so --help and argument errors exit quickly.
from tensorflow.keras.datasets import mnist

model_path = Path(args.model)

with open(model_path, 'r') as file: # load data from --model path
    data = json.load(file)
(inputData, desiredOutputs), (test_X, test_y) = mnist.load_data()
maxRange = 255 # the largest value an input can be, used to normalise inputs and outputs
train_inputs = inputData.reshape(len(inputData), -1).astype(float) / maxRange
train_outputs = np.eye(10)[desiredOutputs]
test_inputs = test_X.reshape(len(test_X), -1).astype(float) / maxRange
test_outputs = np.eye(10)[test_y]

structure = [784,16,16,10] # including input and output neurones
valid = validate_model_data(data, structure)
if args.verbose:
    print(f"Valid: {valid}")

if not valid:
    sys.exit(
        f"Error: {model_path} weights/biases do not match the expected network "
        "structure. Run src/NNrandomiser.py to regenerate the model."
    )
    
L = len(structure)
network = Network(data,structure)
if args.verbose:
    print(network)

# Training
training = args.train # either training or testing
testing = not training

if training:
    print("Training")
    lowestCost = float("inf")
    costs = []
    noOfEpochs = args.epochs
    interval = 5 # what interval to print update message to console to
    #epochs = [i*100 for i in range(int(noOfEpochs/100))]
    learning_rate = args.learning_rate
    for epoch in range(noOfEpochs):  # Train for set number of epochs
        total_cost = 0
        for i, normalised_input in enumerate(train_inputs): # iterates through all training examples
            if i % 10000 == 0: # to show progress every 10,000th training example
                print(f"Training example {i}")
            desired_output = train_outputs[i]
            output = network.forwardPass(normalised_input)
            total_cost += cost(output, desired_output)
            network.backwardPass(desired_output, learning_rate)  # Normalise desired output for backpropagation
        average_cost = total_cost/len(train_inputs)
        costs.append(average_cost)
        if epoch % interval == 0: # every 5th epoch in this case, prints update message
            print(f"Epoch {epoch}, Total Cost: {total_cost}, Average cost/example: {average_cost}, LR: {learning_rate}")
        lowestCost = min(total_cost,lowestCost) 
        
    print(f"Lowest Cost: {lowestCost}, lowest Cost / example: {lowestCost/len(train_inputs)}") # final update message
    if args.save_plot:
        save_cost_plot(costs, args.save_plot)
        print(f"Saved cost plot: {args.save_plot}")

    # Update weights and biases in local memory then model file
    data["weights"] = [weights.tolist() for weights in network.weights]
    data["biases"] = [biases.tolist() for biases in network.biases]
    with open(model_path, 'w') as file:
        json.dump(data, file, indent=4)

if testing: #FIX
    print("Testing")
    totalCost = 0
    noOfExamples = len(test_inputs) # tests on examples set aside to avoid overfitting
    wrong = 0
    for count, normalised_input in enumerate(test_inputs): # all testing examples
        desired = test_outputs[count]
        NNanswer = network.forwardPass(normalised_input)
        prediction = int(np.argmax(NNanswer))
        if prediction != test_y[count]: # increments counter for each incorrect answer
            wrong += 1
        thisCost = cost(desired,NNanswer)
        totalCost += thisCost
        if args.verbose:
            print(f"Given an image, NN returned {prediction}. That should be {test_y[count]}. Cost of that example was {thisCost}")
    correct = noOfExamples - wrong
    print(f"Accuracy: {correct * 100 / noOfExamples:.2f}% ({correct}/{noOfExamples})")
    print(f"Incorrect: {wrong}")
    print(f"Average cost/example: {totalCost/noOfExamples}")
