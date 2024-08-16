This neural network was trained on the MNIST dataset with 60,000 28x28 handwritten digits between 0 and 9 from scratch.

MainNN is the driver code for training and testing the network. 
NNDependencies holds the underlying functions for the network, neurones, activation functions, etc.
data.json holds the weights after 30 epochs of training.
NNrandomiser writes randomised weights and biases for a specified network structure to data.json

This is not optimised and runs solely on the cpu so it's slow as hell
