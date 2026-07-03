# Neural Network from Scratch

This project implements a fully connected feedforward neural network from first principles in Python, without using machine learning frameworks for the network itself.
It uses the MNIST dataset to recognise handwritten digits.

---

## Features

- Matrix-based forward propagation and backpropagation
- Gradient descent weight updates with training support
- Custom sigmoid activation and cost functions
- Configurable network architecture and hyperparameters
- Model saving/loading with JSON

---

## Project Structure

- `mainNN.py` - Training loop and evaluation logic
- `NNDependencies.py` - Core neural network layers and training utilities
- `NNrandomiser.py` - Random weight initialisation module
- `data.json` - Example saved weights
- `requirements.txt` - Python package dependencies

---

## Setup

This project requires Python 3.9 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project:

```bash
python mainNN.py
```

By default, `mainNN.py` loads the saved model data from `data.json`, downloads MNIST through TensorFlow/Keras if needed, and evaluates the network against the test set.

To train instead, set `training = True` in `mainNN.py`.
