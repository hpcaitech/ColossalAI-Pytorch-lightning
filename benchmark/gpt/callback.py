import psutil
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)
    dist.barrier()


def get_cpu_mem():
    return psutil.Process().memory_info().rss


class MemoryMonitor(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.max_cpu_mem = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.max_cpu_mem = max(get_cpu_mem(), self.max_cpu_mem)

    def on_fit_start(self, trainer, pl_module) -> None:
        max_cuda_mem = torch.cuda.max_memory_allocated()
        cuda_mem = torch.cuda.memory_allocated()
        print_rank_0(f'CPU memory before training: {get_cpu_mem()/1024**2:.3f} MB')
        print_rank_0(f'CUDA memory before training: {cuda_mem/1024**2:.3f} MB')
        print_rank_0(f'Max CUDA memory before training: {max_cuda_mem/1024**2:.3f} MB')

    def on_fit_end(self, trainer, pl_module) -> None:
        max_cuda_mem = torch.cuda.max_memory_allocated()
        print_rank_0(f'Max CPU memory: {self.max_cpu_mem/1024**2:.3f} MB')
        print_rank_0(f'Max CUDA memory: {max_cuda_mem/1024**2:.3f} MB')
