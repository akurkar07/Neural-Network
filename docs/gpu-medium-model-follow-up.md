# Medium-Model GPU Benchmark Follow-Up

## Purpose

This follow-up executes the plan in the [CPU and GPU performance report](gpu-performance-report.md): increase the network width, use float32 arrays, keep benchmark data on the selected backend, warm up CuPy, and measure repeated runs.

## Configuration

The benchmark used the full MNIST dataset for one epoch with a Xavier-initialized, seeded `[784, 512, 512, 10]` network. It ran on the same NVIDIA GeForce RTX 3060 and compared NumPy CPU execution with CuPy GPU execution from identical initial weights.

- Train examples: `60,000`
- Test examples: `10,000`
- Learning rate: `0.1`
- Recorded runs per configuration: `3`
- Warm-up batches before each recorded run: `1` for both backends
- Data type: `float32`

![Medium model NumPy and CuPy benchmark](assets/gpu_medium_stats.png)

## Results

| Batch size | NumPy median examples/sec | CuPy median examples/sec | Matched speedup | Matching test accuracy |
|---:|---:|---:|---:|---:|
| 256 | 21,908.39 | 87,919.62 | 4.0x | 41.68% |
| 1024 | 24,267.65 | 308,846.60 | 12.7x | 19.63% |
| 2048 | 22,866.23 | 548,886.08 | 24.0x | 16.36% |
| 4096 | 25,613.13 | 713,056.54 | 27.8x | 12.19% |
| 8192 | 25,688.25 | 742,582.53 | 28.9x | 12.14% |

CuPy was faster at every tested batch size. The best observed warmed throughput was `25,688` examples/sec for NumPy and `742,583` for CuPy, both at batch size `8192`. This is a `28.9x` matched-workload speedup, not a claim of either device's theoretical peak. CPU and GPU runs reached the same final accuracy and cost for every matching configuration, so the difference is a performance crossover rather than a change in training semantics.

At batch size `8192`, median operation end-to-end time was `2.76s` for NumPy and `0.41s` for CuPy, a `6.7x` GPU advantage. This includes model construction, data conversion to the selected backend, training, and final evaluation; it excludes MNIST loading and Python/CUDA process startup. See the [benchmark methodology](benchmark-methodology.md) for the exact timing boundaries and remaining limitations.

## Why the GPU Won

The medium network has roughly `669,000` weights, compared with `12,960` in the original network. The larger dense matrix multiplications provide enough work per CUDA launch for the RTX 3060 to amortize launch overhead and use its parallel capacity. Larger batches reduce the number of updates and launches per epoch, which increases the GPU advantage further.

## Choosing Hardware for Local Models

| Consideration | CPU | GPU |
|---|---|---|
| Small models or small batches | Usually faster because there is no kernel-launch overhead | Often underused; launch and synchronization overhead can dominate |
| Large dense models or large batches | Can become the bottleneck as matrix sizes grow | Usually faster because thousands of cores can work on matrix operations together |
| Interactive single requests | Good choice when low setup latency matters | Useful only when the model is already resident on the GPU or each request is substantial |
| Batch inference or training | Throughput grows modestly with more CPU cores | Strong choice when requests can be batched and GPU memory holds the model and data |
| Memory capacity | Uses system RAM, which is usually larger and easier to expand | Limited by VRAM; model weights, activations, and batches must all fit |
| Memory transfer | No device transfer when data is already in RAM | Host-to-device copies can erase gains if data moves for every request or batch |
| Power and noise | Usually lower for modest local workloads | Often higher, especially under sustained training or inference |
| Setup and portability | Simple NumPy environment; works on most machines | Requires compatible hardware, drivers, CUDA, and a matching CuPy package |
| Cost | Uses hardware already present in most systems | Requires a suitable discrete GPU; cloud GPUs add hourly cost |

Use the CPU for small local models, one-off predictions, development runs, and workloads that cannot batch requests or fit in GPU VRAM. It is simpler, starts quickly, and may be faster, as the original `[784, 16, 16, 10]` benchmark demonstrated.

Use the GPU for training or batch inference when the model has large matrix operations, batches can be made large enough, and weights plus working data remain in VRAM. This medium-model benchmark is the relevant local example: once the workload grew to `[784, 512, 512, 10]`, the GPU delivered up to `22.3x` the CPU throughput at the same batch size.

For local language or image models, load the model once and send multiple requests together where latency requirements permit. Avoid transferring the model or individual inputs between RAM and VRAM for every operation; device residency is often the difference between a GPU win and a GPU slowdown.

## Benchmarking Changes

The implementation now supports the conditions used in this experiment:

- `--structure` configures MNIST-compatible layer sizes.
- `randomiser.py` accepts `--seed` and defaults to Xavier initialization.
- Model parameters, MNIST inputs, targets, activations, and gradients use float32.
- Benchmarks move each dataset to its selected backend once.
- `--benchmark-runs` defaults to three and CuPy warm-up is controlled by `--benchmark-warmup-batches`.
- Summary CSV rows include median, minimum, and maximum duration values.

Raw records are written to `outputs/gpu_medium_peak_summary.csv` and `outputs/gpu_medium_peak_history.csv`. The low one-epoch accuracies should not be compared directly with the earlier five-epoch small-model result; longer accuracy benchmarks should tune the learning rate for each batch-size regime.