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

## GPU Processing

The network supports NumPy for CPU execution and CuPy for CUDA GPU execution. Install the GPU requirements only when using the CuPy backend.

Install a CuPy wheel matching the installed CUDA runtime. For current CUDA 12 systems:

```powershell
pip install -r requirements-gpu.txt
```

Run an individual GPU training or test job with `--backend cupy`:

```powershell
python src/main.py --train --backend cupy --model models/fresh.json --epochs 5 --batch-size 256
```

Benchmark CPU and GPU backends with the same model and settings:

```powershell
python src/main.py --benchmark-batches --backends numpy,cupy --epochs 5 --learning-rate 0.1 --batch-sizes 256 --benchmark-output outputs/numpy_vs_cupy_full_summary.csv --benchmark-history-output outputs/numpy_vs_cupy_full_history.csv --benchmark-plot docs/assets/numpy_vs_cupy_full_stats.png
```

Create a configurable, seeded model:

```powershell
python src/randomiser.py models/medium.json --structure 784,512,512,10 --seed 42
python src/main.py --train --model models/medium.json --structure 784,512,512,10 --backend cupy --epochs 5 --batch-size 256
```

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
- `--backend BACKEND` - Use `numpy` (CPU, default) or `cupy` (CUDA GPU) for training or testing
- `--backends LIST` - Comma-separated backends for `--benchmark-batches`, for example `numpy,cupy`
- `--structure LAYERS` - Comma-separated MNIST layer sizes, for example `784,512,512,10`
- `--model PATH` - Load/save a specific model file
- `--epochs N` - Number of training epochs
- `--learning-rate VALUE` - Training learning rate
- `--batch-size N` - Number of examples per training update
- `--batch-sizes LIST` - Comma-separated batch sizes for benchmarking, for example `1,8,32,128`
- `--cpu-batch-sizes LIST` - Optional NumPy-only batch-size sweep for best-tested CPU throughput
- `--gpu-batch-sizes LIST` - Optional CuPy-only batch-size sweep for best-tested GPU throughput
- `--benchmark-train-limit N` - Limit benchmark training examples for quicker comparisons
- `--benchmark-test-limit N` - Limit benchmark test examples for quicker comparisons
- `--benchmark-runs N` - Recorded runs per backend and batch size; defaults to 3
- `--benchmark-warmup-batches N` - Unrecorded CuPy warm-up batches before each run; defaults to 1
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

## Verification

Run the regression suite, including CPU/GPU output parity when CuPy is installed:

```powershell
python -m unittest discover -s tests -v
```

## Analysis and Results

- [Batch processing analysis](docs/analysis/batch-processing.md)
- [CPU and GPU performance report](docs/analysis/gpu-performance-report.md)
- [Medium-model GPU follow-up](docs/analysis/gpu-medium-model-follow-up.md)
- [Medium-model quality training](docs/analysis/medium-model-quality-training.md)
- [CPU and GPU benchmark methodology](docs/analysis/benchmark-methodology.md)
