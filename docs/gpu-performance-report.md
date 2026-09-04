# CPU and GPU Performance Report

## Result

The full MNIST benchmark trained the same `[784, 16, 16, 10]` model from the same starting weights for 5 epochs, using a learning rate of `0.1` and a batch size of `256`.

| Backend | Hardware | Total time | Examples/sec | Test accuracy | Final average cost |
|---|---|---:|---:|---:|---:|
| NumPy | CPU | 2.11s | 142,103.98 | 92.25% | 0.105871 |
| CuPy | NVIDIA GeForce RTX 3060 | 8.27s | 36,255.72 | 92.25% | 0.105871 |

NumPy was `3.9x` faster overall. The matching cost and accuracy establish that both backends performed the same training calculation; this is a workload-size result, not a correctness problem in the GPU version.

Raw records: `outputs/numpy_vs_cupy_full_summary.csv` and `outputs/numpy_vs_cupy_full_history.csv`.

## Why the CPU Won

The model has only `12,960` trainable weights:

| Layer connection | Weight matrix | Parameters |
|---|---:|---:|
| Input to hidden 1 | `16 x 784` | 12,544 |
| Hidden 1 to hidden 2 | `16 x 16` | 256 |
| Hidden 2 to output | `10 x 16` | 160 |

At batch size `256`, the largest forward matrix multiplication is `16 x 784` by `784 x 256`. This is a small operation for both processors. The CPU's optimized BLAS implementation can complete it with little scheduling overhead.

The GPU has far greater parallel capacity, but it must launch separate kernels for matrix products, sigmoid operations, derivatives, reductions, and gradient updates. Each mini-batch therefore creates many short GPU operations. Kernel launch latency and synchronization cost more than the calculation saves when the matrices are this small.

The per-epoch records also show a GPU warm-up effect. NumPy training stayed near `0.38s` per epoch. CuPy took `2.61s` in epoch one, then about `0.71s` per epoch. The first epoch includes CUDA library/context initialization; even after warm-up, the GPU's small-operation overhead remains larger than the CPU time.

The current full benchmark also evaluates the test set after every epoch. This creates additional GPU work and host-to-device data preparation in the measured total time. It is valid for an end-to-end comparison, but it is not a pure training-throughput measurement.

## What Must Change for a GPU Advantage

The target is not custom CUDA C++. The target is enough arithmetic per GPU launch to amortize overhead while keeping data resident on the device.

### 1. Increase Model Width

Make the architecture configurable and compare progressively larger hidden layers. Good initial experiments are:

| Experiment | Structure | Approximate weights |
|---|---|---:|
| Current baseline | `[784, 16, 16, 10]` | 13 thousand |
| Medium | `[784, 512, 512, 10]` | 669 thousand |
| Large | `[784, 1024, 1024, 10]` | 1.85 million |

The larger matrices will provide substantially more parallel work per batch. Use identical structures, initialization seeds, epochs, learning rates, and batch sizes for both backends.

### 2. Increase Batch Size

Test `256`, `512`, `1024`, and `2048` examples per update, subject to GPU memory. Larger batches turn each matrix multiply into a wider operation and reduce the number of Python loop iterations and CUDA launches per epoch.

Batch size also changes optimization behavior. Report accuracy and cost beside throughput rather than treating the fastest size as automatically best.

### 3. Use Float32 End to End

The project currently creates `float64` arrays. Convert loaded MNIST data, model parameters, activations, and gradients to `float32`. Consumer GPUs are optimized heavily for single-precision work, and float32 halves memory traffic and GPU memory use.

Numerical parity should then be evaluated with a tolerance rather than exact equality. The CPU and GPU should still reach comparable cost and accuracy.

### 4. Keep Benchmark Data on the GPU

Transfer training and test datasets to the selected backend once before timing. Reuse the GPU-resident test arrays for each epoch's evaluation. Keep model saving and CSV/plot conversion outside the timed region.

Report two metrics:

- End-to-end time: includes dataset transfer, CUDA initialization, training, and evaluation.
- Steady-state training time: excludes one-time setup and measures repeated training epochs after warm-up.

Both metrics matter. End-to-end time answers whether a short one-off run benefits; steady-state time answers whether a long training job benefits.

### 5. Warm Up and Repeat

Before a timed CuPy run, execute one unrecorded forward/backward batch and synchronize the CUDA stream. Run each configuration at least three times, then report median time and the run-to-run range. CUDA clocks, first-use library loading, and operating-system activity otherwise make short measurements noisy.

## Success Criteria

The next benchmark should use the medium structure, float32 arrays, GPU-resident datasets, and batch sizes from `256` to `2048`. The GPU path has an advantage only when it meets both conditions:

1. Its median steady-state examples/sec is greater than NumPy's for the same configuration.
2. Its final test accuracy remains within an agreed tolerance of the NumPy result.

If the medium model does not cross over, repeat with the large model before considering lower-level optimization. Only consider custom CuPy fused kernels after profiling identifies an element-wise operation as a significant cost; CUDA C++ is not justified unless profiling shows library calls are no longer the dominant work.