import torch.nn as nn
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from data import build_data
from timm.models.vision_transformer import _create_vision_transformer, _cfg
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.lr_scheduler import LinearWarmupLR
from strategies import ColossalAIStrategy


def vit_cifar(**kwargs):
    pretrained_cfg = _cfg(num_classes=10, input_size=(3, 32, 32), crop_pct=1.0)
    model_kwargs = dict(patch_size=4, embed_dim=512, depth=6, num_heads=8,
                        drop_rate=0.1, mlp_ratio=1.0, **kwargs)
    model = _create_vision_transformer('vit_cifar', pretrained_cfg=pretrained_cfg, **model_kwargs)
    return model


class Cifar10PlModule(pl.LightningModule):
    def __init__(self, warmup_epochs: int, lr: float = 1e-3, adjust_lr_by_step: bool = False) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.adjust_lr_by_step = adjust_lr_by_step
        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.top1_accuracy = Accuracy(top_k=1)

    def configure_sharded_model(self) -> None:
        self.model = vit_cifar()

    def configure_optimizers(self):
        opt = HybridAdam(self.model.parameters(), self.lr)
        total_steps = self.trainer.max_epochs
        warmup_steps = self.warmup_epochs
        interval = 'epoch'
        if self.adjust_lr_by_step:
            total_steps *= self.trainer.estimated_stepping_batches
            warmup_steps *= self.trainer.estimated_stepping_batches
            interval = 'step'
        scheduler = LinearWarmupLR(opt, total_steps, warmup_steps)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': interval}}

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        self.top1_accuracy(F.log_softmax(logits, dim=1), labels)
        self.log_dict({'valid_loss': loss, 'valid_acc': self.top1_accuracy},
                      prog_bar=True, on_epoch=True, sync_dist=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--adjust_lr_by_step', action='store_true', default=False)
    parser.add_argument('--colossal', action='store_true', default=False)
    args = parser.parse_args()
    assert args.batch_size % args.np == 0
    seed_everything(args.seed)
    batch_size_per_dp = args.batch_size // args.np
    trainer_cfg = {
        'accelerator': 'cuda',
        'strategy': 'ddp'
    }
    if args.colossal:
        trainer_cfg = {
            'strategy': ColossalAIStrategy(
                use_chunk=True,
                enable_distributed_storage=True,
                placement_policy='cuda'
            )
        }
    trainer = pl.Trainer(devices=args.np, max_epochs=args.epochs,
                         callbacks=[TQDMProgressBar()], **trainer_cfg)
    trainloader, testloader = build_data(batch_size_per_dp)
    model = Cifar10PlModule(args.warmup, args.lr, args.adjust_lr_by_step)
    trainer.fit(model, trainloader, testloader)
