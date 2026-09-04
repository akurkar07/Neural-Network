# Medium-Model GPU Benchmark Follow-Up

## Purpose

This follow-up executes the plan in the [CPU and GPU performance report](gpu-performance-report.md): increase network width, use float32 arrays, keep benchmark data on the selected backend, warm up both backends, and measure repeated runs.

## Configuration

The benchmark used the full MNIST dataset for one epoch with a Xavier-initialised, seeded `[784, 512, 512, 10]` network. It ran on the NVIDIA GeForce RTX 3060 and compared NumPy CPU execution with CuPy GPU execution from identical initial weights.

- Train examples: `60,000`
- Test examples: `10,000`
- Learning rate: `0.1`
- Recorded runs per configuration: `3`
- Warm-up batches before each recorded run: `1` for both backends
- Data type: `float32`

![Medium model NumPy and CuPy benchmark](../assets/gpu_medium_stats.png)

## Results

| Batch size | NumPy median examples/sec | CuPy median examples/sec | Matched speedup | Matching test accuracy |
|---:|---:|---:|---:|---:|
| 256 | 21,908.39 | 87,919.62 | 4.0x | 41.68% |
| 1024 | 24,267.65 | 308,846.60 | 12.7x | 19.63% |
| 2048 | 22,866.23 | 548,886.08 | 24.0x | 16.36% |
| 4096 | 25,613.13 | 713,056.54 | 27.8x | 12.19% |
| 8192 | 25,688.25 | 742,582.53 | 28.9x | 12.14% |

CuPy was faster at every tested batch size. The best observed warmed throughput was `25,688` examples/sec for NumPy and `742,583` for CuPy, both at batch size `8192`. This is a `28.9x` matched-workload speedup within the tested range, not a claim of either device's theoretical peak.

At batch size `8192`, median operation end-to-end time was `2.76s` for NumPy and `0.41s` for CuPy, a `6.7x` GPU advantage. This includes model construction, selected-backend data conversion, training, and final evaluation; it excludes MNIST loading and Python/CUDA process startup.

## Local Model Trade-offs

| Consideration | CPU | GPU |
|---|---|---|
| Small models or batches | Usually faster due to low launch overhead | Often underused; launch overhead can dominate |
| Large dense models or batches | Can become the bottleneck | Usually faster when VRAM holds model and data |
| Interactive single requests | Good for low setup latency | Good when model is already resident on the device |
| Batch inference or training | Throughput grows modestly with core count | Strong throughput when work can be batched |
| Memory | Uses system RAM | Limited by VRAM |
| Data movement | No device transfer for RAM-resident data | Repeated RAM-to-VRAM transfers can erase gains |
| Setup | Portable NumPy environment | Requires compatible hardware, drivers, CUDA, and CuPy |

Use CPU execution for small local models, one-off predictions, and workloads that cannot batch requests or fit in VRAM. Use GPU execution for training or batched inference when the model has large matrix operations and remains on the GPU.

See the [benchmark methodology](benchmark-methodology.md) for the exact timing boundaries and remaining limitations.