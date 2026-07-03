import numpy as np
def sigmoid(x:np.ndarray):
    """Does sigmoid on each element in a NumPy array."""
    return 1 / (1 + np.exp(-x))  
 
def sigmoid_derivative(x:np.ndarray):
    """Returns the derivative of the sigmoid function at x."""
    sig = sigmoid(x)
    return sig * (1 - sig)

def cost(y:np.ndarray,i:np.ndarray) -> np.int64:
    "Returns the sum of the squares of the differences between output from forward pass (y) and target (i)"
    return np.sum((y-i)**2)

def numToList(input:int) -> np.ndarray:
    """turns a desired numerical output into an output list to be used to calculate a cost\n
    eg. 7 -> [0,0,0,0,0,0,0,1,0,0,0]"""
    out = np.array([0]*10)
    out[input] = 1
    return out

class Network:
    """Represents a fully connected feedforward neural network."""
    def __init__(self,data,structure:list) -> None:
        """Builds the network layers from saved weights, biases, and structure."""
        self.structure = structure  # Number of neurons in each layer
        self.L = len(structure)

        self.weights = [np.array(layer, dtype=float) for layer in data['weights']]
        # Biases are reshaped into column vectors so W @ activation + bias stays (layer_size, 1).
        self.biases = [np.array(layer, dtype=float).reshape(-1,1) for layer in data["biases"]]
    
    def forwardPass(self, normalisedInputs: np.ndarray):
        """
        Takes a normalised list of inputs and runs a matrix-based forward pass.
        Returns normalised result.
        """
        # Inputs are reshaped into a column vector to match matrix multiplication dimensions.
        activation = np.array(normalisedInputs, dtype=float).reshape(-1,1)

        self.activations = [activation]
        self.z_values = []

        for weights, bias in zip(self.weights, self.biases):
            z = weights @ activation + bias # @ is numpy's matrix multiply operator since * is element-wise
            activation = sigmoid(z)

            self.z_values.append(z)
            self.activations.append(activation)

        return activation.flatten()

    def backwardPass(self,normalisedOutputs:np.ndarray,learningRate):
        """
        Calculates gradients of output layer, then backpropagates error through layers until first hidden layer
        """
        if not hasattr(self, "activations") or not hasattr(self, "z_values"):
            raise RuntimeError("forwardPass must be called before backwardPass.")

        desired_outputs = np.array(normalisedOutputs, dtype=float).reshape(-1,1)
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        # Output layer gradient for squared error cost.
        error = 2 * (self.activations[-1] - desired_outputs) * sigmoid_derivative(self.z_values[-1])
        weight_gradients[-1] = error @ self.activations[-2].T
        bias_gradients[-1] = error

        # Hidden layer gradients, moving backwards through the network.
        for layer in range(len(self.weights) - 2, -1, -1):
            error = (self.weights[layer + 1].T @ error) * sigmoid_derivative(self.z_values[layer])
            weight_gradients[layer] = error @ self.activations[layer].T
            bias_gradients[layer] = error

        self.weights = [
            weights - learningRate * gradient
            for weights, gradient in zip(self.weights, weight_gradients)
        ]
        self.biases = [
            bias - learningRate * gradient
            for bias, gradient in zip(self.biases, bias_gradients)
        ]

    def __repr__(self) -> str:
        """Returns a readable summary of the number of neurones in each layer."""
        lines = []
        for count, size in enumerate(self.structure):
            lines.append(f"Layer {count}: {size} neurones.")

        lines.append("Weight matrices:")
        for count, weights in enumerate(self.weights):
            lines.append(f"W{count + 1}: {weights.shape}")

        return "NETWORK: \n"+"\n".join(lines) + "\n------------"
