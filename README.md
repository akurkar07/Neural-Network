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

Run the project in testing mode:

```bash
python mainNN.py --test
```

Show every test prediction:

```bash
python mainNN.py --test --verbose
```

TensorFlow startup logs are hidden by default. Show them with:

```bash
python mainNN.py --test --show-tf-logs
```

Run the project in training mode:

```bash
python mainNN.py --train
```

Training defaults to 30 epochs and a learning rate of 0.1. Override them with:

```bash
python mainNN.py --train --epochs 10 --learning-rate 0.05
```

`mainNN.py` loads the saved model data from `data.json`, downloads MNIST through TensorFlow/Keras if needed, and then runs the selected mode.
