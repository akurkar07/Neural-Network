# Neural Network from Scratch

This project implements a fully-connected feedforward neural network from first principles in Python, without using machine learning frameworks.  
It was trained on the MNIST dataset to recognise handwritten digits.

---

## Features
- Matrix-based forward propagation and backpropagation
- Gradient descent weight updates with minibatch training
- Custom activation functions (sigmoid, ReLU, softmax)
- Configurable network architecture and hyperparameters
- Model saving/loading (JSON)

---

## Project Structure
- `mainNN.py` — Training loop and evaluation logic  
- `NNDependencies.py` — Core neural network layers and training utilities  
- `NNrandomiser.py` — Random weight initialisation module  
- `data.json` — Example saved weights  
- `outputs/` — Example training outputs (accuracy, loss)

---

## Usage
1. Clone the repository  
2. Install dependencies (Python ≥ 3.9, `numpy`, `matplotlib`)  
3. Run:
   ```bash
   python mainNN.py
