import json
import numpy as np
structure = [784,16,16,10]

with open("data.json", 'r') as file:
    data = json.load(file)
with open("data.json", 'w') as file:
    file.truncate()
    data["weights"] = [np.random.randn(y, x).tolist() for x, y in zip(structure[:-1], structure[1:])]
    data["biases"] = [np.random.randn(y, 1).tolist() for y in structure[1:]]
    json.dump(data, file, indent=4)
