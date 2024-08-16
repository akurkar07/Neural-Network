import random,json
from NNDependencies import *
from keras._tf_keras.keras.datasets import mnist

with open("data.json", 'r') as file:
    data = json.load(file)
(inputData, desiredOutputs), (test_X, test_y) = mnist.load_data()
maxRange = 255 # the largest value an input can be, used to normalise inputs and outputs

structure = [784,16,16,10] # including input and output neurones
valid = True
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
    training = False
    testing = not training

    if training:
        print("Training")
        lowestCost = float("inf")
        costs = []
        noOfEpochs = 30
        #epochs = [i*100 for i in range(int(noOfEpochs/100))]
        learning_rate = data["learningRate"]
        for epoch in range(noOfEpochs):  # Train for 30000 epochs
            total_cost = 0
            for i, image in enumerate(inputData):
                if i % 10000 == 0:
                    print(f"Training example {i}")
                desired_output = numToList(desiredOutputs[i])
                normalised_input = np.array(image).flatten()
                normalised_input = [i/255 for i in normalised_input]
                output = network.forwardPass(normalised_input)
                total_cost += cost(output, desired_output)
                network.backwardPass(desired_output, learning_rate)  # Normalise desired output for backpropagation
            if epoch % 1 == 0:
                costs.append(total_cost)
                print(f"Epoch {epoch}, Total Cost: {total_cost}, Average cost/example: {total_cost/len(inputData)}, LR: {learning_rate}")
            lowestCost = min(total_cost,lowestCost)
            
        print(f"Lowest Cost: {lowestCost}, lowest Cost / example: {lowestCost/len(inputData)}")

        # Update weights and biases in the JSON data
        data["weights"] = [[neurone.weights for neurone in layer] for layer in network.network[1:]]
        data["biases"] = [[neurone.bias for neurone in layer] for layer in network.network[1:]]
        with open("data.json", 'w') as file:
            json.dump(data, file, indent=4)

    if testing: #FIX
        print("TESTING")
        totalCost = 0
        noOfExamples = len(test_X)
        wrong = 0
        for count, image in enumerate(test_X): # first 1000 testing examples
            desired = numToList(test_y[count])
            normalised_input = np.array(image).flatten()
            normalised_input = [i/255 for i in normalised_input]
            NNanswer = network.forwardPass(normalised_input)
            if NNanswer.index(max(NNanswer)) != test_y[count]:
                wrong += 1
            thisCost = cost(desired,NNanswer)
            totalCost += thisCost
            print(f"Given an image, NN returned {NNanswer.index(max(NNanswer))}. That should be {test_y[count]}. Cost of that example was {thisCost}")
        print(f"Average cost/example was {totalCost/noOfExamples}, no. of incorrectly identified examples: {wrong}\nPercentage identified {(noOfExamples-wrong)*100/noOfExamples}%")
