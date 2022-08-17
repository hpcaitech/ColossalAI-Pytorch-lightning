# ColossalAI Pytorch-lightning strategy

## Usage

```python
from pytorch_lightning.strategies.colossalai import ColossalAIStrategy

trainer = Trainer(..., strategy=ColossalAIStrategy())
```