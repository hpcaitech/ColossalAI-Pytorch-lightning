# Train ViT on CIFAR-10

## Run

Prepare dataset:
```shell
export DATA=/path/to/cifar10
```

Torch DDP:
```shell
python train.py --np 4
```

ColossalAI ZeRO-DP:
```shell
python train.py --np 4 --colossal
```

## Result

| Strategy | GPUs |  Validation loss | Validation Accuracy |
| --- | --- | --- | --- |
| ddp | 4 | 0.650 | 0.834 |
| colossalai | 4 | 0.651 | 0.833 |