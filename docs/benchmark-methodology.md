# CPU and GPU Benchmark Methodology

## What the Measurements Answer

The project reports two different questions. They should not be treated as interchangeable.

| Question | Measurement | What it supports |
|---|---|---|
| Which backend is faster for the same work? | Same model, seed, data, batch size, epochs, and learning rate | A matched-workload speedup claim |
| Which backend has the best throughput in the tested range? | Sweep batch sizes independently for NumPy and CuPy | A best-tested peak-throughput comparison |
| Which backend completes a local job sooner? | `end_to_end_seconds` | A practical operation-time comparison |
| Which backend sustains training throughput? | `total_seconds` and `median_examples_per_second` after warm-up | A steady-state comparison |
| Which setup learns a task sooner? | Time to a target validation accuracy with tuned learning rate | A training-quality comparison; not yet implemented |

The first two questions are performance comparisons. They do not establish that the GPU is always better for local models or that one configuration is an absolute hardware peak.

## Timing Boundaries

Every recorded run exposes both timings:

| Field | Included | Excluded |
|---|---|---|
| `total_seconds` | Training and per-epoch evaluation after warm-up | Model creation and conversion of NumPy data to the selected backend |
| `end_to_end_seconds` | Model creation, selected-backend conversion, training, and final evaluation | MNIST loading through TensorFlow and Python/CUDA process startup |

CuPy operations are synchronized before timing values are read. Without synchronization, a host timer can stop while CUDA work is only queued, producing an invalid GPU result.

## Controls for Fairness

- The same saved model starts every CPU and GPU run.
- Both backends receive the same number of unrecorded warm-up batches before each recorded run.
- Benchmarks record three runs by default and emit median, minimum, and maximum durations.
- Matching batch sizes permit direct CPU-to-GPU speedup claims.
- `--cpu-batch-sizes` and `--gpu-batch-sizes` permit independent sweeps to find each backend's best tested result.
- Dataset loading is outside the benchmark because TensorFlow loading is not work performed by either network backend.

## Remaining Limits

`peak_median_examples_per_second` means the best result within the supplied batch-size list, not a hardware manufacturer's theoretical peak. A stronger peak study should also control CPU BLAS thread settings, CPU power mode, GPU clocks, background load, and a wider range of batch sizes up to memory limits.

Large batches change the number of gradient updates and can change convergence. Do not compare time-to-accuracy across different batch sizes with a fixed learning rate. Tune the learning rate and epoch budget for each batch-size regime, then compare the time required to reach the same validation target.

## Recommended Commands

Use a common sweep for direct same-batch comparisons:

```powershell
python src/randomiser.py models/medium.json --structure 784,512,512,10 --seed 42
python src/main.py --benchmark-batches --model models/medium.json --structure 784,512,512,10 --backends numpy,cupy --batch-sizes 256,1024,2048,4096,8192 --epochs 1 --benchmark-runs 3
```

Use independent ranges for a best-tested peak comparison:

```powershell
python src/main.py --benchmark-batches --model models/medium.json --structure 784,512,512,10 --backends numpy,cupy --cpu-batch-sizes 256,1024,2048,4096 --gpu-batch-sizes 512,1024,2048,4096,8192 --epochs 1 --benchmark-runs 3
```

Interpret the generated CSV as follows:

- Compare `median_examples_per_second` at a shared batch size for matched-workload speedup.
- Compare `peak_median_examples_per_second` for the best result observed in each backend's supplied sweep.
- Compare `median_end_to_end_seconds` when deciding whether a local job finishes sooner.