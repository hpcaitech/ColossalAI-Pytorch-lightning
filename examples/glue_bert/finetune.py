from datetime import datetime
from typing import Optional

import datasets
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import (
    AutoConfig,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification
)
from data import GLUEDataModule
from argparse import ArgumentParser
from colossalai.nn.optimizer import HybridAdam
from pytorch_lightning.strategies.colossalai import ColossalAIStrategy


class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_fraction: float = 0.0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def configure_sharded_model(self) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path, config=self.config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = HybridAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                               eps=self.hparams.adam_epsilon)
        num_warmup_steps = int(self.trainer.estimated_stepping_batches * self.hparams.warmup_fraction)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', default='mrpc')
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2.4e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_fraction', type=float, default=0.1)
    parser.add_argument('--colossal', action='store_true', default=False)
    args = parser.parse_args()
    assert args.batch_size % args.np == 0
    local_batch_size = args.batch_size // args.np
    seed_everything(42)
    model_name = 'bert-base-uncased'
    dm = GLUEDataModule(
        model_name_or_path=model_name,
        task_name=args.task,
        train_batch_size=local_batch_size,
        eval_batch_size=local_batch_size
    )
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        train_batch_size=local_batch_size,
        eval_batch_size=local_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
    )
    trainer_cfg = {
        'accelerator': 'cuda',
        'strategy': 'ddp',
    }
    if args.colossal:
        trainer_cfg = {
            'strategy': ColossalAIStrategy(
                use_chunk=True,
                enable_distributed_storage=True,
                placement_policy='cuda',
                initial_scale=32
            )
        }
    trainer = Trainer(
        max_epochs=args.epochs,
        devices=args.np,
        **trainer_cfg
    )
    trainer.fit(model, datamodule=dm)
