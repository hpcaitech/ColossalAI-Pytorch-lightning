import pytorch_lightning as pl
import argparse
from strategies import ColossalAIStrategy
from data import RandomDataloader
from model import GPTLitModule
from callback import MemoryMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model', default='gpt2_xl')
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--no_activation_ckpt', action='store_true', default=False)
    parser.add_argument('--opt_nvme_offload_frac', type=float, default=0.0)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--colossal', action='store_true', default=False)
    parser.add_argument('--placement_policy', default='cuda')
    parser.add_argument('--opt_gpu_margin_rat', type=float, default=0.0)
    args = parser.parse_args()
    model = GPTLitModule(args.model, checkpoint=not args.no_activation_ckpt,
                         optimizer_nvme_offload_fraction=args.opt_nvme_offload_frac)
    train_dataloader = RandomDataloader(args.steps_per_epoch, args.batch_size, args.seq_len)
    trainer_cfg = {
        'accelerator': 'cuda',
        'precision': 16,
        'strategy': DDPStrategy(static_graph=True)
    }
    if args.colossal:
        trainer_cfg = {
            'strategy': ColossalAIStrategy(
                placement_policy=args.placement_policy,
                gpu_margin_mem_ratio=args.opt_gpu_margin_rat,
                initial_scale=32
            )
        }
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.np,
        enable_checkpointing=False,
        callbacks=[MemoryMonitor()],
        **trainer_cfg
    )
    trainer.fit(model, train_dataloader)
