import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import GPT2Config, GPT2LMHeadModel
from colossalai.nn.optimizer import HybridAdam
from typing import Callable
from torch.optim import Adam
from strategies import ColossalAIStrategy


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


def gpt_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def gpt2_tiny(checkpoint=True):
    return GPTLMModel(hidden_size=128, num_layers=4, num_attention_heads=4, checkpoint=checkpoint)


def gpt2_medium(checkpoint=True):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


class RandomDataloader:
    def __init__(self, n_steps: int, batch_size: int, seq_len: int = 1024, vocab_size: int = 50257) -> None:
        self.n_steps = n_steps
        self.cur_step = 0
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        self.cur_step = 0
        return self

    def __next__(self):
        if self.cur_step >= self.n_steps:
            raise StopIteration
        self.cur_step += 1
        return get_data(self.batch_size, self.seq_len, self.vocab_size)


class GPTPretrain(pl.LightningModule):
    def __init__(self, gpt_init_func: Callable[[bool], nn.Module], checkpoint: bool = True) -> None:
        super().__init__()
        self.gpt_init_func = gpt_init_func
        self.checkpoint = checkpoint
        self.criterion = gpt_lm_loss
        self.model = self.gpt_init_func(self.checkpoint)

    def configure_sharded_model(self) -> None:
        self.model = self.gpt_init_func(self.checkpoint)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, input_ids)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return HybridAdam(self.model.parameters(), lr=1e-3)


if __name__ == '__main__':
    gpt_pretrain = GPTPretrain(gpt2_tiny)
    train_dataloader = RandomDataloader(10, 2)
    trainer = pl.Trainer(max_epochs=5, devices=4, strategy=ColossalAIStrategy(placement_policy='auto'))
    trainer.fit(gpt_pretrain, train_dataloader)
