# Finetune BERT on GLUE

## Run

Torch DDP:
```shell
python finetune.py
```

ColossalAI ZeRO-DP:
```shell
python finetune.py --colossal
```

## Result

| Strategy | GPUs |  Validation loss | Validation Accuracy | Validation F1 |
| --- | --- | --- | --- | --- |
| ddp | 1 | 0.365 | 0.853 | 0.896 |
| colossalai | 1 | 0.358 | 0.863 | 0.902 |
| ddp | 2 | 0.375 | 0.848 | 0.893 |
| colossalai | 2 | 0.368 | 0.848 | 0.892 |
| ddp | 4 | 0.415 | 0.850 | 0.894 |
| colossalai | 4 | 0.389 | 0.848 | 0.892 |