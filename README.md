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

- `src/mainNN.py` - Training loop and evaluation logic
- `src/NNDependencies.py` - Core neural network layers and training utilities
- `src/NNrandomiser.py` - Random weight initialisation module
- `data.json` - Example saved weights
- `requirements.txt` - Python package dependencies

---

## Quickstart

Copy and paste this into PowerShell from the project folder:

```powershell
# Create and activate a local virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create, train, and test a fresh random model
python src/NNrandomiser.py models/fresh.json
python src/mainNN.py --train --model models/fresh.json --epochs 1 --learning-rate 0.1
python src/mainNN.py --test --model models/fresh.json
```

To test the included saved model only:

```powershell
python src/mainNN.py --test
```

---

## Usage

### `mainNN.py`

Train or test a saved model file. By default, this uses `data.json`.
Training saves updated weights and biases back to the selected model file.

```bash
python src/mainNN.py --test
python src/mainNN.py --train
```

Training defaults to 30 epochs and a learning rate of 0.1:

```bash
python src/mainNN.py --train --epochs 10 --learning-rate 0.05
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
python src/NNrandomiser.py models/fresh.json
```

Existing model files are protected by default. To overwrite one, pass `--force` and confirm the prompt:

```bash
python src/NNrandomiser.py models/fresh.json --force
```

The scripts download MNIST through TensorFlow/Keras if needed. TensorFlow startup logs are hidden by default during `mainNN.py` runs.
