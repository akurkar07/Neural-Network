import math
import numpy as np
def sigmoid(input):
    return 1 / (1 + math.exp(-input))  
 
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def cost(outputs:list,intendedNumbers:list) -> float:
    "Returns the sum of the squares of the differences between data and target"
    return sum([(o - i)**2 for o,i in zip(outputs, intendedNumbers)])

def numToList(input):
    """turns a desired numerical output into an output list to be used to calculate a cost\n
    eg. 7 -> [0,0,0,0,0,0,0,1,0,0,0]"""
    a = [0]*10
    a[input] = 1
    return a

class Network:
    def __init__(self,data,structure:list) -> None:
        self.structure = structure  # Number of neurons in each layer
        self.L = len(structure)
        self.network = []
        for layer in range(self.L): # creates the network structure in form [layer,layer,layer] where each layer is in form [neurone,neurone,neurone]
            a = []
            for num in range(self.structure[layer]):
                if layer != 0:
                    bias = data["biases"][layer-1][num] # writes weights and biases to neurone as it is created
                    weights = data["weights"][layer-1][num]
                    a.append(Neuron(layer,num,weights,bias))
                else:
                    a.append(Neuron(layer,num))
            self.network.append(a)
    
    def forwardPass(self, normalisedInputs: list):
        """        
        Takes a normalised list of inputs, assigns them to the input neurones and does a forward pass.\n
        Returns normalised result.
        """
        fromPrevLayer = normalisedInputs
        forNextLayer = []
        for layer in self.network:
            for neurone in layer:
                # each neurone gets all the activations from the previous layer, appending its own activation to the list to be fed into the next layer
                forNextLayer.append(neurone.feedforward(fromPrevLayer)) 
            fromPrevLayer = forNextLayer[:] # shallow copy of the data being passed forward to avoid it being edited
            forNextLayer = []
        return fromPrevLayer

    def backwardPass(self,normalisedOutputs:list,learningRate):
        """
        Calculates gradients of output layer, then backpropagates error through layers until first hidden layer
        """

        fromSubsequentLayer = [normalisedOutputs,0] # these are normalised desired values with the 0 meaning no subsequent derivatives
        forNextLayer = [[],[]]
        for count, layer in enumerate(reversed(self.network[1:])):  # Excluding input neurons
            #before getting the errors of the neurones, compile the activations of the next layer to calculate the gradients for weights
            nextActivations = [neurone.activation for neurone in self.network[self.L-count-2]]
            for neurone in layer: # compiles a list of the errors and weights of each layer whilst using that data from the layer after it
                error = neurone.learn(fromSubsequentLayer,nextActivations,learningRate)
                forNextLayer[0].append(error)
                forNextLayer[1].append(neurone.weights)
            fromSubsequentLayer = [forNextLayer[0], np.array(forNextLayer[1][:]).transpose().tolist()] # weight matrix transposed to enable dot product
            forNextLayer = [[], []]

    def __repr__(self) -> str:
        lst = []
        for count, layer in enumerate(self.network):
            lst.append(f"Layer {count}: {len(layer)} neurones.")
        return "NETWORK: \n"+"\n".join(lst) + "\n------------"

class Neuron:
    def __init__(self,layer,num,weights=0,bias=[0]) -> None:
        self.layer = layer
        self.num = num # which neurone in the layer it is
        self.label = f"|Layer {self.layer}, Number {self.num}| "
        if type(bias) != float: # not ideal
            self.bias = bias[0]
        else:
            self.bias = bias
        self.weights = weights

    def feedforward(self, inputs:list):
        if self.layer == 0: # input neurones have no weights or biases so their activation is just whatever input they receive
            self.activation = inputs[self.num]
        else:
            self.z = sum([x * w for x,w in zip(inputs, self.weights)]) + self.bias # dot product of previous activations and weights, plus bias
            self.activation = sigmoid(self.z) #activation function applied here
        return self.activation

    def learn(self,inputs:list,nextActivations,learningRate):
        """returns only the error of a neurone for now.\n
        Inputs will either be normalised desired values or the errors of the subsequent layer's neurones\n
        Inputs formatted [derivatives,subsequentWeights]"""
        derivatives = inputs[0] 
        d_a_d_z = sigmoid_derivative(self.z) # derivative of activation w.r.t z for layer L
        if inputs[1] == 0: 
            error = d_a_d_z * 2 * (self.activation-derivatives[self.num]) # not really derivatives, more like desireds
        else:# the sum of the products of the weights attached from this neurone to all subsequent neurones and their bias gradients
            subsequentWeights = inputs[1][self.num if self.num < len(inputs[1]) else -1]
            error =  d_a_d_z* sum([w * x for w,x in zip(subsequentWeights, derivatives)])  # dot product of the derivatives and the weights from L+1
        self.error = error
        self.bias -= learningRate*error # the neurone's weights and bias is updated
        self.weights = [weight-learningRate*error*nextActivations[count] for count,weight in enumerate(self.weights)]

        return error

    def __repr__(self) -> str:
        return self.label
