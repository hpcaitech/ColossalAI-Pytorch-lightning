from data import DaliImagenetDataModule
from loss import MulticlassBCEWithLogitsLoss, MixupLoss, LabelSmoothLoss
from timm.models.vision_transformer import vit_base_patch16_224, vit_small_patch16_224
from pytorch_lightning import seed_everything
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import torch
import os
from argparse import ArgumentParser


class ViTModule(pl.LightningModule):
    def __init__(self, mixup: bool = True, schedule_lr_by_step: bool = False) -> None:
        super().__init__()
        self.mixup = mixup
        self.schedule_lr_by_step = schedule_lr_by_step
        self.criterion = MulticlassBCEWithLogitsLoss(0.1)
        if mixup:
            self.criterion = MixupLoss(self.criterion)
        self.acc = Accuracy()

    def configure_sharded_model(self) -> None:
        self.model = vit_small_patch16_224(drop_rate=0.1, weight_init='jax', num_classes=100)

    def configure_optimizers(self):
        opt = HybridAdam(self.model.parameters(), lr=3e-3, weight_decay=0.3)
        if self.schedule_lr_by_step:
            num_steps = self.trainer.estimated_stepping_batches
            interval = 'step'
        else:
            num_steps = self.trainer.max_epochs
            interval = 'epoch'
        num_warmups = int(num_steps * 0.1)
        scheduler = CosineAnnealingWarmupLR(opt, num_steps, num_warmups)
        scheduler = {'scheduler': scheduler, 'interval': interval}
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs)
        if self.mixup:
            loss = self.criterion(logits, **targets)
        else:
            loss = self.criterion(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.mixup:
            if targets['targets_a'].ndim == 0:
                targets['targets_a'] = targets['targets_a'].unsqueeze(0)
            if targets['targets_b'].ndim == 0:
                targets['targets_b'] = targets['targets_b'].unsqueeze(0)
        logits = self.model(inputs)
        if self.mixup:
            loss = self.criterion(logits, **targets)
        else:
            loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, 1)
        if self.mixup:
            self.acc(preds, targets['targets_a'])
        else:
            self.acc(preds, targets)
        self.log_dict({'valid_loss': loss, 'valid_acc': self.acc}, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_load_checkpoint(self, checkpoint) -> None:
        if not hasattr(self, 'model'):
            self.configure_sharded_model()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--schedule_lr_by_step', action='store_true', default=False)
    args = parser.parse_args()
    assert args.batch_size % args.np == 0
    local_batch_size = args.batch_size // args.np
    print(f'local batch size: {local_batch_size}, total batch size: {args.batch_size}')
    seed_everything(42)
    dm = DaliImagenetDataModule(os.environ['DATA'], local_batch_size,
                                mixup_alpha=0.2, randaug_magnitude=10, randaug_num_layers=2,
                                gpu_aug=True)
    model = ViTModule(mixup=True, schedule_lr_by_step=args.schedule_lr_by_step)
    trainer_cfg = {
        'accelerator': 'cuda',
        'strategy': 'ddp',
    }
    trainer = pl.Trainer(
        max_epochs=300,
        devices=args.np,
        gradient_clip_val=1.0,
        resume_from_checkpoint=args.ckpt,
        callbacks=[LearningRateMonitor('step')],
        **trainer_cfg
    )
    trainer.fit(model, datamodule=dm)
