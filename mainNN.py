import argparse
import json
import os
import sys

import numpy as np

from NNDependencies import Network, cost, numToList

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
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0.")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0.")

    return args

def validate_model_data(data, structure):
    """Checks saved weights and biases match the expected network structure."""
    # These checks make sure the dimensions of the json's data are present and correct.
    if "weights" not in data or "biases" not in data:
        return False

    # Checks no. of layers in JSON matches structure.
    if len(data["weights"]) != len(structure[1:]) or len(data["biases"]) != len(structure[1:]):
        return False

    # Checks no. of neurones in each layer matches structure.
    for check in ("weights","biases"):
        for count, layer in enumerate(data[check]):
            if len(layer) != structure[1:][count]:
                return False

    return True

args = parse_args()

if not args.show_tf_logs: # Hides the tensorflow logs if --show-tf-logs isn't enabled
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# TensorFlow is imported after CLI parsing so --help and argument errors exit quickly.
from tensorflow.keras.datasets import mnist

with open("data.json", 'r') as file: # load data
    data = json.load(file)
(inputData, desiredOutputs), (test_X, test_y) = mnist.load_data()
maxRange = 255 # the largest value an input can be, used to normalise inputs and outputs

structure = [784,16,16,10] # including input and output neurones
valid = validate_model_data(data, structure)
if args.verbose:
    print(f"Valid: {valid}")

if not valid:
    sys.exit(
        "Error: data.json weights/biases do not match the expected network "
        "structure. Run NNrandomiser.py to regenerate them or update data.json."
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
        for i, image in enumerate(inputData): # iterates through all training examples
            if i % 10000 == 0: # to show progress every 10,000th training example
                print(f"Training example {i}")
            desired_output = numToList(desiredOutputs[i])
            normalised_input = np.array(image).flatten()
            normalised_input = [i/255 for i in normalised_input]
            output = network.forwardPass(normalised_input)
            total_cost += cost(output, desired_output)
            network.backwardPass(desired_output, learning_rate)  # Normalise desired output for backpropagation
        if epoch % interval == 0: # every 5th epoch in this case, prints update message
            costs.append(total_cost)
            print(f"Epoch {epoch}, Total Cost: {total_cost}, Average cost/example: {total_cost/len(inputData)}, LR: {learning_rate}")
        lowestCost = min(total_cost,lowestCost) 
        
    print(f"Lowest Cost: {lowestCost}, lowest Cost / example: {lowestCost/len(inputData)}") # final update message

    # Update weights and biases in local memory then data.json
    data["weights"] = [[neurone.weights for neurone in layer] for layer in network.network[1:]]
    data["biases"] = [[neurone.bias for neurone in layer] for layer in network.network[1:]]
    with open("data.json", 'w') as file:
        json.dump(data, file, indent=4)

if testing: #FIX
    print("Testing")
    totalCost = 0
    noOfExamples = len(test_X) # trains on all testing examples set aside to avoid overfitting
    wrong = 0
    for count, image in enumerate(test_X): # all testing examples
        desired = numToList(test_y[count])
        normalised_input = np.array(image).flatten() #turns 2d array 1d
        normalised_input = [i/255 for i in normalised_input]
        NNanswer = network.forwardPass(normalised_input)
        if NNanswer.index(max(NNanswer)) != test_y[count]: # increments counter for each incorrect answer
            wrong += 1
        thisCost = cost(desired,NNanswer)
        totalCost += thisCost
        if args.verbose:
            print(f"Given an image, NN returned {NNanswer.index(max(NNanswer))}. That should be {test_y[count]}. Cost of that example was {thisCost}")
    correct = noOfExamples - wrong
    print(f"Accuracy: {correct * 100 / noOfExamples:.2f}% ({correct}/{noOfExamples})")
    print(f"Incorrect: {wrong}")
    print(f"Average cost/example: {totalCost/noOfExamples}")
