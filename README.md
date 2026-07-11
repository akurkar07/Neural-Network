# Neural Network from Scratch

This project implements a fully connected feedforward neural network from first principles in Python, without using machine learning frameworks for the network itself.
It uses the MNIST dataset to recognise handwritten digits.

I built this as a learning tool to understand matrix-based gradient descent from scratch. The first version was written in 2023 completely blind, using only 3Blue1Brown's maths tutorial videos on neural networks and gradient descent as guidance, especially [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w).

---

## Features

- Matrix-based forward propagation and backpropagation
- Gradient descent weight updates with training support
- Custom sigmoid activation and cost functions
- Configurable network architecture and hyperparameters
- Model saving/loading with JSON

---

## Design Notes

The core network is intentionally written around arrays and matrix operations:

- each layer stores a weight matrix and bias vector
- forward propagation uses `weights @ activation + bias`
- backpropagation uses matrix products, transposes, element-wise sigmoid derivatives, and gradient updates

That makes the project a good candidate for a GPU version with CuPy. In principle, most NumPy calls in the core math can be swapped for CuPy equivalents because CuPy mirrors much of the NumPy API and runs those array operations on the GPU.

A GPU version should still be designed carefully around data movement. The important rule is to keep arrays on the GPU during training instead of repeatedly converting between NumPy arrays and CuPy arrays. The algorithm does not need to be redesigned around manual parallelism; the matrix operations already expose parallel work, and CuPy delegates that work to CUDA kernels. The main redesign is therefore an array-backend layer, for example choosing `numpy` or `cupy` as `xp`, plus explicit conversion when loading JSON, saving JSON, or interacting with TensorFlow/Keras data.

---

## Project Structure

- `src/main.py` - Command-line entry point for training, testing, and benchmarking
- `src/dependencies.py` - Core neural network layers and matrix operations
- `src/data.py` - Model validation, model JSON loading/saving, and MNIST loading
- `src/training.py` - Shared training, evaluation, batching, and metric plotting helpers
- `src/benchmark.py` - Batch-size benchmark recording and graph generation
- `src/randomiser.py` - Random weight initialisation module
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
python src/randomiser.py models/fresh.json
python src/main.py --train --model models/fresh.json --epochs 1 --learning-rate 0.1
python src/main.py --test --model models/fresh.json
```

To test the included saved model only:

```powershell
python src/main.py --test
```

---

## Usage

### `main.py`

Train or test a saved model file. By default, this uses `data.json`.
Training saves updated weights and biases back to the selected model file.

```bash
python src/main.py --test
python src/main.py --train
```

Training defaults to 30 epochs and a learning rate of 0.1:

```bash
python src/main.py --train --epochs 10 --learning-rate 0.05
```

Useful flags:

- `--test` - Evaluate a model against the MNIST test set
- `--train` - Train a model and save updated weights/biases
- `--benchmark-batches` - Compare training cost, timing, throughput, and test accuracy across batch sizes
- `--model PATH` - Load/save a specific model file
- `--epochs N` - Number of training epochs
- `--learning-rate VALUE` - Training learning rate
- `--batch-size N` - Number of examples per training update
- `--batch-sizes LIST` - Comma-separated batch sizes for benchmarking, for example `1,8,32,128`
- `--benchmark-train-limit N` - Limit benchmark training examples for quicker comparisons
- `--benchmark-test-limit N` - Limit benchmark test examples for quicker comparisons
- `--benchmark-output PATH` - Save benchmark summary stats as CSV
- `--benchmark-history-output PATH` - Save per-epoch benchmark stats as CSV
- `--benchmark-plot PATH` - Save benchmark comparison graphs
- `--verbose` - Print model details and every test prediction
- `--show-tf-logs` - Show TensorFlow startup logs

Train with mini-batches:

```bash
python src/main.py --train --model models/fresh.json --epochs 5 --learning-rate 0.1 --batch-size 32
```

Benchmark different batch sizes from the same starting model:

```bash
python src/main.py --benchmark-batches --model models/fresh.json --epochs 1 --batch-sizes 1,8,16,32,64,128
```

By default, benchmark results are saved to:

- `outputs/batch_benchmark_summary.csv`
- `outputs/batch_benchmark_history.csv`
- `outputs/batch_benchmark_stats.png`

For a quick benchmark while experimenting, limit the dataset:

```bash
python src/main.py --benchmark-batches --model models/fresh.json --epochs 1 --batch-sizes 1,16,64,256 --benchmark-train-limit 5000 --benchmark-test-limit 1000
```

### `randomiser.py`

Create a new random model file:

```bash
python src/randomiser.py models/fresh.json
```

Existing model files are protected by default. To overwrite one, pass `--force` and confirm the prompt:

```bash
python src/randomiser.py models/fresh.json --force
```

The scripts download MNIST through TensorFlow/Keras if needed. TensorFlow startup logs are hidden by default during `main.py` runs.
