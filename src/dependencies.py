import numpy as np


def get_backend(name):
    """Returns the requested array module."""
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError(
                "The CuPy backend requires CuPy. Install a CUDA-compatible CuPy package first."
            ) from error
        return cp
    raise ValueError(f"Unsupported backend: {name}")


def sigmoid(x, xp=np):
    """Does sigmoid on each element in an array."""
    return 1 / (1 + xp.exp(-x))
 
def sigmoid_derivative(x, xp=np):
    """Returns the derivative of the sigmoid function at x."""
    sig = sigmoid(x, xp)
    return sig * (1 - sig)

def cost(y, i, xp=np):
    "Returns the sum of the squares of the differences between output from forward pass (y) and target (i)"
    return xp.sum((y-i)**2)

def numToList(input:int) -> np.ndarray:
    """turns a desired numerical output into an output list to be used to calculate a cost\n
    eg. 7 -> [0,0,0,0,0,0,0,1,0,0,0]"""
    out = np.array([0]*10)
    out[input] = 1
    return out

class Network:
    """Represents a fully connected feedforward neural network."""
    def __init__(self, data, structure: list, backend="numpy") -> None:
        """Builds the network layers from saved weights, biases, and structure."""
        self.structure = structure  # Number of neurons in each layer
        self.L = len(structure)
        self.backend = backend
        self.xp = get_backend(backend)

        self.weights = [self.xp.array(layer, dtype=float) for layer in data['weights']]
        # Biases are reshaped into column vectors so W @ activation + bias stays (layer_size, 1).
        self.biases = [self.xp.array(layer, dtype=float).reshape(-1, 1) for layer in data["biases"]]
    
    def _forward_columns(self, activation: np.ndarray):
        """
        Runs a forward pass where each column is one training example.
        """
        self.activations = [activation]
        self.z_values = []

        for weights, bias in zip(self.weights, self.biases):
            z = weights @ activation + bias # @ is numpy's matrix multiply operator since * is element-wise
            activation = sigmoid(z, self.xp)

            self.z_values.append(z)
            self.activations.append(activation)

        return activation

    def forwardPass(self, normalisedInputs: np.ndarray):
        """
        Takes a normalised list of inputs and runs a matrix-based forward pass.
        Returns normalised result.
        """
        # Inputs are reshaped into a column vector to match matrix multiplication dimensions.
        activation = self.xp.array(normalisedInputs, dtype=float).reshape(-1, 1)
        activation = self._forward_columns(activation)

        return activation.flatten()

    def forwardBatch(self, normalisedInputs: np.ndarray):
        """
        Takes a batch of normalised inputs and returns one output row per example.
        """
        # Batch rows become columns so layer math stays W @ A + b.
        activation = self.xp.array(normalisedInputs, dtype=float).T
        activation = self._forward_columns(activation)

        return activation.T

    def backwardPass(self,normalisedOutputs:np.ndarray,learningRate):
        """
        Calculates gradients of output layer, then backpropagates error through layers until first hidden layer
        """
        self.backwardBatch(self.xp.array(normalisedOutputs, dtype=float).reshape(1, -1), learningRate)

    def backwardBatch(self, normalisedOutputs:np.ndarray, learningRate):
        """
        Calculates average gradients for a batch, then updates weights and biases.
        """
        if not hasattr(self, "activations") or not hasattr(self, "z_values"):
            raise RuntimeError("forwardPass must be called before backwardPass.")

        desired_outputs = self.xp.array(normalisedOutputs, dtype=float).T
        batch_size = desired_outputs.shape[1]
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        # Output layer gradient for squared error cost.
        error = 2 * (self.activations[-1] - desired_outputs) * sigmoid_derivative(self.z_values[-1], self.xp)
        weight_gradients[-1] = (error @ self.activations[-2].T) / batch_size
        bias_gradients[-1] = self.xp.mean(error, axis=1, keepdims=True)

        # Hidden layer gradients, moving backwards through the network.
        for layer in range(len(self.weights) - 2, -1, -1):
            error = (self.weights[layer + 1].T @ error) * sigmoid_derivative(self.z_values[layer], self.xp)
            weight_gradients[layer] = (error @ self.activations[layer].T) / batch_size
            bias_gradients[layer] = self.xp.mean(error, axis=1, keepdims=True)

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
