import random,json
from NNDependencies import *
from keras._tf_keras.keras.datasets import mnist

with open("data.json", 'r') as file: # load data
    data = json.load(file)
(inputData, desiredOutputs), (test_X, test_y) = mnist.load_data()
maxRange = 255 # the largest value an input can be, used to normalise inputs and outputs

structure = [784,16,16,10] # including input and output neurones
valid = True # these checks make sure the dimensions of the json's data are present and correct
if (len(data["weights"]) == len(data["biases"]) == len(structure[1:])) == False: # checks no. of layers in JSON matches structure
    valid = False
for check in ("weights","biases"): # checks no. of neurones in each layer matches structure
    for count, layer in enumerate(data[check]):
        if len(layer) != structure[1:][count]:
            valid = False
print(f"Valid: {valid}")

if valid:
    L = len(structure)
    network = Network(data,structure)
    print(network)

# Training
    training = False # either training or testing
    testing = not training

    if training:
        print("Training")
        lowestCost = float("inf")
        costs = []
        noOfEpochs = 30
        interval = 5 # what interval to print update message to console to
        #epochs = [i*100 for i in range(int(noOfEpochs/100))]
        learning_rate = data["learningRate"]
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
        print("TESTING")
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
            print(f"Given an image, NN returned {NNanswer.index(max(NNanswer))}. That should be {test_y[count]}. Cost of that example was {thisCost}")
        print(f"Average cost/example was {totalCost/noOfExamples}, no. of incorrectly identified examples: {wrong}\nPercentage identified {(noOfExamples-wrong)*100/noOfExamples}%")
