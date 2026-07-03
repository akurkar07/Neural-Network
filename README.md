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

## Quickstart

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

Test the included saved model:

```bash
python mainNN.py --test
```

Train the included saved model:

```bash
python mainNN.py --train --epochs 10 --learning-rate 0.05
```

Create a fresh random model file:

```bash
python NNrandomiser.py models/fresh.json
```

Train and test that model:

```bash
python mainNN.py --train --model models/fresh.json
python mainNN.py --test --model models/fresh.json
```

---

## Usage

### `mainNN.py`

Train or test a saved model file. By default, this uses `data.json`.
Training saves updated weights and biases back to the selected model file.

```bash
python mainNN.py --test
python mainNN.py --train
```

Training defaults to 30 epochs and a learning rate of 0.1:

```bash
python mainNN.py --train --epochs 10 --learning-rate 0.05
```

Useful flags:

- `--test` - Evaluate a model against the MNIST test set
- `--train` - Train a model and save updated weights/biases
- `--model PATH` - Load/save a specific model file
- `--epochs N` - Number of training epochs
- `--learning-rate VALUE` - Training learning rate
- `--verbose` - Print model details and every test prediction
- `--show-tf-logs` - Show TensorFlow startup logs

### `NNrandomiser.py`

Create a new random model file:

```bash
python NNrandomiser.py models/fresh.json
```

Existing model files are protected by default. To overwrite one, pass `--force` and confirm the prompt:

```bash
python NNrandomiser.py models/fresh.json --force
```

The scripts download MNIST through TensorFlow/Keras if needed. TensorFlow startup logs are hidden by default during `mainNN.py` runs.
