# Benchmark Results

## Model scaling

RAM: 500G

GPU: A100 (40G)

We fix the batch size per GPU to 1.

| Strategy | GPUs | Max model size |
| deepspeed (zero3 offload) | 1 |  |
| colssalai (auto) | 1 |  |
| deepspeed (zero3 offload) | 8 |  |
| colssalai (auto) | 8 |  |

## Small model comparison

We collected results using GPT2-XL (~1.6B) that fit training with DDP (AMP).

All experiments are run on 4x A100 (40G).

| Strategy | Global batch size | Global throughput (samples/sec) | Max CUDA memory allocated (MB) |
| --- | --- | --- | --- |
| ddp (AMP) | 24 | 11.76 | 37905.422 |
| deepspeed (zero3) | 160 | 18.18 | 25360.968 |
| colssalai (cuda) | 160 | 19.36 | 24003.394 |
