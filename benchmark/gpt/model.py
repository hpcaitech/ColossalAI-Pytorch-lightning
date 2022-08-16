import torch.nn as nn
import pytorch_lightning as pl
from transformers import GPT2Config, GPT2LMHeadModel
from colossalai.nn.optimizer import HybridAdam

__all__ = ['GPTLitModule']


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


def gpt2_tiny(checkpoint=True):
    return GPTLMModel(hidden_size=128, num_layers=4, num_attention_heads=4, checkpoint=checkpoint)


def gpt2_small(checkpoint=True):
    return GPTLMModel(hidden_size=768, num_layers=12, num_attention_heads=12, checkpoint=checkpoint)


def gpt2_medium(checkpoint=True):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_large(checkpoint=True):
    return GPTLMModel(hidden_size=1280, num_layers=36, num_attention_heads=20, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=25, checkpoint=checkpoint)


def gpt2_2B(checkpoint=True):
    return GPTLMModel(hidden_size=2048, num_layers=40, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_3B(checkpoint=True):
    return GPTLMModel(hidden_size=2304, num_layers=48, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_4B(checkpoint=True):
    return GPTLMModel(hidden_size=2304, num_layers=64, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_6B(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_8B(checkpoint=True):
    return GPTLMModel(hidden_size=3072, num_layers=72, num_attention_heads=24, checkpoint=checkpoint)


def gpt2_12B(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=60, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_15B(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=78, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_18B(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=90, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_20B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=25, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_24B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_28B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=35, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_32B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=40, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_36B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=45, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_40B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_45B(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=56, num_attention_heads=16, checkpoint=checkpoint)


def gpt3(checkpoint=True):
    return GPTLMModel(max_seq_len=2048, hidden_size=12288, num_layers=96, num_attention_heads=96, checkpoint=checkpoint)


def get_gpt_model(model_name: str, checkpoint: bool = True) -> nn.Module:
    model_map = {
        'gpt2_tiny': gpt2_tiny,
        'gpt2_small': gpt2_small,
        'gpt2_medium': gpt2_medium,
        'gpt2_large': gpt2_large,
        'gpt2_xl': gpt2_xl,
        'gpt2_2B': gpt2_2B,
        'gpt2_3B': gpt2_3B,
        'gpt2_4B': gpt2_4B,
        'gpt2_6B': gpt2_6B,
        'gpt2_8B': gpt2_8B,
        'gpt2_12B': gpt2_12B,
        'gpt2_15B': gpt2_15B,
        'gpt2_18B': gpt2_18B,
        'gpt2_20B': gpt2_20B,
        'gpt2_24B': gpt2_24B,
        'gpt2_28B': gpt2_28B,
        'gpt2_32B': gpt2_32B,
        'gpt2_36B': gpt2_36B,
        'gpt2_40B': gpt2_40B,
        'gpt2_45B': gpt2_45B,
        'gpt3': gpt3,
    }
    assert model_name in model_map
    return model_map[model_name](checkpoint)


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class GPTLitModule(pl.LightningModule):
    def __init__(self, model_name: str, checkpoint: bool = True, optimizer_nvme_offload_fraction: float = 0.0) -> None:
        super().__init__()
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.optimizer_nvme_offload_fraction = optimizer_nvme_offload_fraction
        self.criterion = GPTLMLoss()

    def configure_sharded_model(self) -> None:
        self.model = get_gpt_model(self.model_name, self.checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:
        if not hasattr(self, 'model'):
            self.configure_sharded_model()

    def configure_optimizers(self):
        return HybridAdam(self.model.parameters(), lr=1e-3, nvme_offload_dir='./offload', nvme_offload_fraction=self.optimizer_nvme_offload_fraction)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, input_ids)
        return loss
