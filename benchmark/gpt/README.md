# Benchmark Results

## Model scaling

RAM: 500G

GPU: A100 (40G)

We fix the batch size per GPU to 1.

| Strategy | GPUs | Max model size (B) |  Max CUDA memory allocated (MB) | Step time (sec) |
| --- | --- | --- | --- | --- |
| deepspeed (zero3 offload) | 1 | 18 | 5699.500 | 39.32 |
| colssalai (auto) | 1 | 24 | 36483.311 | 69.54 |
| deepspeed (zero3) | 8 | 12 | 29751.203 | 9.06 |
| colssalai (cuda) | 8 | 12 | 24504.032 | 7.07 |

Commands:

Deepspeed:
```shell
python train.py --epochs 1 --steps_per_epoch 3 --model gpt2_18B --strategy deepspeed --offload
python train.py --epochs 1 --steps_per_epoch 3 --model gpt2_12B --strategy deepspeed --np 8
```

ColossalAI:
```shell
python train.py --epochs 1 --steps_per_epoch 3 --model gpt2_24B --strategy colossal --placement_policy auto --opt_gpu_margin_rat 0.9
python train.py --epochs 1 --steps_per_epoch 3 --model gpt2_12B --strategy colossal --np 8
```

## Small model comparison

We collected results using GPT2-XL (~1.6B) that fit training with DDP (AMP).

All experiments are run on 4x A100 (40G).

| Strategy | Global batch size | Global throughput (samples/sec) | Max CUDA memory allocated (MB) |
| --- | --- | --- | --- |
| ddp (AMP) | 24 | 11.76 | 37905.422 |
| deepspeed (zero3) | 160 | 18.18 | 25360.968 |
| colssalai (cuda) | 160 | 19.36 | 24003.394 |

Commands:

DDP:
```shell
python train.py --np 4 --batch_size 6
```

Deepspeed:
```shell
python train.py --np 4 --strategy deepspeed --batch_size 40
```

ColossalAI:
```shell
python train.py --np 4 --strategy colossal --batch_size 40
```