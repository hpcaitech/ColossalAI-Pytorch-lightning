import pytorch_lightning as pl
import argparse
from data import RandomDataloader
from model import GPTLitModule, get_optimizer
from callback import MemoryMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.strategies.colossalai import ColossalAIStrategy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', default='gpt2_xl')
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--no_activation_ckpt', action='store_true', default=False)
    parser.add_argument('--opt_nvme_offload_frac', type=float, default=0.0)
    parser.add_argument('--opt_nvme_offload_dir', default='./offload')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--placement_policy', default='cuda')
    parser.add_argument('--opt_gpu_margin_rat', type=float, default=0.0)
    parser.add_argument('--cuda_mem_frac', type=float, default=1.0)
    parser.add_argument('--strategy', default='ddp', choices=['ddp', 'colossal', 'deepspeed'])
    parser.add_argument('--offload', action='store_true', default=False)
    args = parser.parse_args()
    train_dataloader = RandomDataloader(args.steps_per_epoch, args.batch_size, args.seq_len)
    optimizer_cfg = {'lr': args.lr}
    if args.strategy == 'ddp':
        trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': DDPStrategy(static_graph=True)
        }
    elif args.strategy == 'colossal':
        trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': ColossalAIStrategy(
                placement_policy=args.placement_policy,
                gpu_margin_mem_ratio=args.opt_gpu_margin_rat,
                initial_scale=32
            )
        }
        optimizer_cfg['nvme_offload_dir'] = args.opt_nvme_offload_dir
        optimizer_cfg['nvme_offload_fraction'] = args.opt_nvme_offload_frac
    elif args.strategy == 'deepspeed':
        trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': DeepSpeedStrategy(
                stage=3,
                offload_parameters=args.offload,
                offload_optimizer=args.offload,
                initial_scale_power=5
            )
        }
        optimizer_cfg['offload'] = args.offload
    opt_init_fn = get_optimizer(args.strategy, **optimizer_cfg)
    model = GPTLitModule(args.model, opt_init_fn, checkpoint=not args.no_activation_ckpt,
                         cuda_mem_fraction=args.cuda_mem_frac)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.np,
        enable_checkpointing=False,
        callbacks=[MemoryMonitor()],
        **trainer_cfg
    )
    trainer.fit(model, train_dataloader)
