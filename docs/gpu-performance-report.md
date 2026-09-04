# CPU and GPU Performance Report

## Result

The full MNIST benchmark trained the same `[784, 16, 16, 10]` model from the same starting weights for 5 epochs, using a learning rate of `0.1` and a batch size of `256`.

| Backend | Hardware | Total time | Examples/sec | Test accuracy | Final average cost |
|---|---|---:|---:|---:|---:|
| NumPy | CPU | 2.11s | 142,103.98 | 92.25% | 0.105871 |
| CuPy | NVIDIA GeForce RTX 3060 | 8.27s | 36,255.72 | 92.25% | 0.105871 |

NumPy was `3.9x` faster overall. The matching cost and accuracy establish that both backends performed the same training calculation; this is a workload-size result, not a correctness problem in the GPU version.

Raw records: `outputs/numpy_vs_cupy_full_summary.csv` and `outputs/numpy_vs_cupy_full_history.csv`.

## Medium Model Follow-Up

The first planned crossover experiment used a Xavier-initialized `[784, 512, 512, 10]` model, float32 arrays, GPU-resident benchmark data, and three warmed runs per configuration. It trained the full MNIST dataset for one epoch on the same NVIDIA GeForce RTX 3060.

| Batch size | NumPy median examples/sec | CuPy median examples/sec | CuPy advantage | Matching test accuracy |
|---:|---:|---:|---:|---:|
| 256 | 19,434.68 | 72,329.84 | 3.7x | 41.68% |
| 1024 | 20,771.66 | 294,824.94 | 14.2x | 19.63% |
| 2048 | 21,818.94 | 490,936.90 | 22.5x | 16.36% |

The GPU is now faster at every tested batch size. This confirms the original diagnosis: once the network is wide enough to create larger dense matrix operations, the RTX 3060 can amortize CUDA launch overhead and use its parallel capacity effectively. The identical accuracy for each backend shows that the result is a performance crossover, not a change in training semantics.

The low one-epoch accuracies are expected with the selected large batch sizes and should not be compared to the earlier five-epoch small-model result. Future quality comparisons should use the same epoch budget and tune the learning rate for each batch-size regime.

Raw records: `outputs/gpu_medium_summary.csv` and `outputs/gpu_medium_history.csv`. Each summary row now includes median, minimum, and maximum duration values for its backend/batch-size configuration.

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

Implemented: the `--structure` option makes architectures configurable, and `randomiser.py` can create reproducible seeded models. Good follow-up experiments are:

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

Implemented: MNIST data, model parameters, activations, and gradients use `float32`. Consumer GPUs are optimized heavily for single-precision work, and float32 halves memory traffic and GPU memory use.

Numerical parity should then be evaluated with a tolerance rather than exact equality. The CPU and GPU should still reach comparable cost and accuracy.

### 4. Keep Benchmark Data on the GPU

Implemented for benchmarks: training and test datasets transfer to the selected backend once before timing and are reused for each epoch's evaluation. Model saving and CSV/plot conversion remain outside the timed region.

Report two metrics:

- End-to-end time: includes dataset transfer, CUDA initialization, training, and evaluation.
- Steady-state training time: excludes one-time setup and measures repeated training epochs after warm-up.

Both metrics matter. End-to-end time answers whether a short one-off run benefits; steady-state time answers whether a long training job benefits.

### 5. Warm Up and Repeat

Implemented: before each timed CuPy run, an unrecorded forward/backward batch initializes backend libraries on a disposable network. The default is three recorded runs per configuration, and summary CSV rows include median, minimum, and maximum duration values.

## Success Criteria

The medium-model benchmark satisfies the performance crossover criteria:

1. CuPy's median examples/sec is greater than NumPy's for every tested batch size.
2. Each CPU/GPU pair reaches matching final test accuracy and cost for the same configuration.

The next work is longer quality benchmarks using an appropriate learning-rate schedule for each batch size, followed by the large `[784, 1024, 1024, 10]` model. Only consider custom CuPy fused kernels after profiling identifies an element-wise operation as a significant cost; CUDA C++ is not justified unless profiling shows library calls are no longer the dominant work.