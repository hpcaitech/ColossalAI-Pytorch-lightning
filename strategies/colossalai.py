import torch
import pytorch_lightning as pl
import contextlib
from typing import Optional, Generator, Any
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
from colossalai.nn.optimizer import CPUAdam, HybridAdam
from pytorch_lightning.strategies.ddp import DDPStrategy
from precision_plugins import ColossalAIPrecisionPlugin
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc


class ModelShardedContext(ColoInitContext):
    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        super()._post_init_method(module, *args, **kwargs)
        module._colossalai_module = True


class ColossalAIStrategy(DDPStrategy):

    def __init__(self, use_chunk: bool = True, chunk_size: Optional[int] = None, enable_distributed_storage: bool = True, placement_policy: str = 'auto', force_outputs_fp32: bool = False) -> None:
        accelerator = CUDAAccelerator()
        precision_plugin = ColossalAIPrecisionPlugin()
        super().__init__(accelerator=accelerator, precision_plugin=precision_plugin)
        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.enable_distributed_storage = enable_distributed_storage
        self.placement_policy = placement_policy
        self.force_outputs_fp32 = force_outputs_fp32
        self._num_nodes = 1
        self._logger = get_dist_logger()

    def setup_distributed(self):
        disable_existing_loggers()
        gpc.init_global_dist(rank=self.global_rank, world_size=self.world_size, backend='nccl',
                             host=self.cluster_environment.main_address, port=self.cluster_environment.main_port)
        gpc.set_device(self.local_rank)
        self.process_group = ProcessGroup()

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly, which is useful for extremely large models which can save memory and
        initialization time.

        Returns: Model parallel context.
        """
        with ModelShardedContext():
            yield

    def setup_precision_plugin(self) -> None:
        super().setup_precision_plugin()
        assert len(self.optimizers) == 1, 'ColossalAIStrategy only supports single Optimizer now.'
        optimizer = self.optimizers[0]
        assert isinstance(optimizer, (CPUAdam, HybridAdam)
                          ), 'ColossalAIStrategy only supports colossalai.nn.optimizer.CPUAdam and colossalai.nn.optimizer.HybridAdam now'
        if self.use_chunk:
            chunk_size = self.chunk_size or ChunkManager.search_chunk_size(self.model, 64 * 1024**2, 1024)
        else:
            chunk_size = None
        chunk_manager = ChunkManager(chunk_size, self.process_group, self.enable_distributed_storage,
                                     GeminiManager.get_default_device(self.placement_policy))
        gemini_manager = GeminiManager(self.placement_policy, chunk_manager)
        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        model = _LightningModuleWrapperBase(self.model)
        self.model = ZeroDDP(model, gemini_manager, self.force_outputs_fp32)
        self.optimizers = [ZeroOptimizer(optimizer, self.model, initial_scale=32)]

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        self.lightning_module._device = self.root_device
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        self.model_to_device()

    @property
    def root_device(self) -> torch.device:
        if self.parallel_devices is not None:
            return self.parallel_devices[self.local_rank]
        return get_current_device()

    def model_to_device(self) -> None:
        pl_module = self.lightning_module
        pl_module._device = self.root_device
        for child in pl_module.modules():
            if child is not pl_module and getattr(child, '_colossalai_module', None) is not True:
                child.to(self.root_device)

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        if isinstance(self.model, ZeroDDP):
            return unwrap_lightning_module(self.model.module)
        return super().lightning_module

    def teardown(self) -> None:
        pass

    def optimizer_step(self, optimizer, opt_idx: int, closure, model=None, **kwargs: Any) -> Any:
        model = model or self.model
        return self.precision_plugin.optimizer_step(model, optimizer, opt_idx, closure, **kwargs)

    def lightning_module_state_dict(self):
        return self.model.state_dict()

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return True
