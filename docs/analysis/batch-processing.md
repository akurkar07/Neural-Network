# Batch Processing Analysis

## Purpose

The original training loop updated the model after every image. Mini-batch processing instead groups examples into a matrix and applies one averaged gradient update per batch.

For batch size `16`, sixteen `(784,)` inputs become a `(784, 16)` activation matrix internally. The forward and backward equations are unchanged; matrix operations process the columns together.

## Full MNIST Result

This benchmark used `60,000` training examples, `10,000` test examples, 5 epochs, and a learning rate of `0.1`.

![Batch size benchmark comparing per-image processing with batch size 16](../assets/batch_1_vs_16_full_stats.png)

| Batch size | Processing style | Weight updates | Total time | Examples/sec | Test accuracy |
|---:|---|---:|---:|---:|---:|
| 1 | Per-image update | 300,000 | 83.96s | 3,573.20 | 92.59% |
| 16 | Mini-batch update | 18,750 | 5.68s | 52,850.24 | 92.40% |

Batch size `16` was `14.8x` faster while retaining nearly identical accuracy. Fewer, larger matrix operations reduce Python-loop overhead and are also a better fit for GPU execution. The per-image method performed more frequent updates and reached slightly higher accuracy in this run, at a much higher time cost.