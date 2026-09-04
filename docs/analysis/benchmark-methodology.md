# CPU and GPU Benchmark Methodology

## Summary

This project reports matched-workload speedup and best-tested throughput as separate results. It records warmed steady-state time and practical operation time separately, runs each configuration three times by default, and uses medians rather than a single favourable timing.

## What the Measurements Answer

| Question | Measurement | What it supports |
|---|---|---|
| Which backend is faster for the same work? | Same model, seed, data, batch size, epochs, and learning rate | A matched-workload speedup claim |
| Which backend has the best throughput in the tested range? | Sweep batch sizes independently for NumPy and CuPy | A best-tested peak-throughput comparison |
| Which backend completes a local job sooner? | `end_to_end_seconds` | A practical operation-time comparison |
| Which backend sustains training throughput? | `total_seconds` and `median_examples_per_second` after warm-up | A steady-state comparison |
| Which setup learns a task sooner? | Time to a target validation accuracy with tuned learning rate | Not yet implemented |

## Timing Boundaries

| Field | Included | Excluded |
|---|---|---|
| `total_seconds` | Training and per-epoch evaluation after warm-up | Model creation and data conversion to the selected backend |
| `end_to_end_seconds` | Model creation, selected-backend conversion, training, and final evaluation | MNIST loading and Python/CUDA process startup |

CuPy operations are synchronised before timing values are read. Without synchronisation, a host timer can stop while CUDA work is only queued.

## Controls and Limits

- The same saved model starts every CPU and GPU run.
- Both backends receive the same number of unrecorded warm-up batches.
- Benchmarks record three runs by default and emit median, minimum, and maximum durations.
- Matching batch sizes permit direct CPU-to-GPU speedup claims.
- `--cpu-batch-sizes` and `--gpu-batch-sizes` permit independent best-tested sweeps.

`peak_median_examples_per_second` is the best result within the supplied batch-size list, not a hardware manufacturer's theoretical peak. Large batches change gradient-update counts and may change convergence, so compare time-to-accuracy only after tuning learning rates and epoch budgets per batch-size regime.

## Commands

Use a common sweep for matched-workload comparisons:

```powershell
python src/randomiser.py models/medium.json --structure 784,512,512,10 --seed 42
python src/main.py --benchmark-batches --model models/medium.json --structure 784,512,512,10 --backends numpy,cupy --batch-sizes 256,1024,2048,4096,8192 --epochs 1 --benchmark-runs 3
```

Use independent ranges for a best-tested comparison:

```powershell
python src/main.py --benchmark-batches --model models/medium.json --structure 784,512,512,10 --backends numpy,cupy --cpu-batch-sizes 256,1024,2048,4096 --gpu-batch-sizes 512,1024,2048,4096,8192 --epochs 1 --benchmark-runs 3
```