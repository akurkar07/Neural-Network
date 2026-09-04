# CPU and GPU Performance Report

## Result

The full MNIST benchmark trained the same `[784, 16, 16, 10]` model from the same starting weights for 5 epochs, using a learning rate of `0.1` and a batch size of `256`.

| Backend | Hardware | Total time | Examples/sec | Test accuracy | Final average cost |
|---|---|---:|---:|---:|---:|
| NumPy | CPU | 2.11s | 142,103.98 | 92.25% | 0.105871 |
| CuPy | NVIDIA GeForce RTX 3060 | 8.27s | 36,255.72 | 92.25% | 0.105871 |

NumPy was `3.9x` faster overall. The matching cost and accuracy establish that both backends performed the same training calculation; this is a workload-size result, not a correctness problem in the GPU version.

The planned medium-model experiment was completed successfully. See the [medium-model GPU follow-up](gpu-medium-model-follow-up.md) for its results.

## Why the CPU Won

The model has only `12,960` trainable weights. At batch size `256`, its largest forward matrix multiplication is `16 x 784` by `784 x 256`, a small operation for optimised CPU BLAS.

The GPU must launch separate kernels for matrix products, sigmoid operations, derivatives, reductions, and gradient updates. For this small network, kernel-launch latency and synchronisation cost more than the calculation saves. The first CuPy epoch also includes CUDA context and library initialisation.

## Path to a GPU Advantage

The completed follow-up applied these steps:

1. Increase model width to create larger dense matrix operations.
2. Increase batch size to reduce update and launch counts.
3. Use float32 for parameters, activations, gradients, and MNIST data.
4. Keep benchmark data on the selected backend.
5. Warm up both backends and report repeated-run medians.

See the [benchmark methodology](benchmark-methodology.md) for timing boundaries and fair-comparison rules.