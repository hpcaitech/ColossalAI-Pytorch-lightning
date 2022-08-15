import os
import glob
from pytorch_lightning import LightningDataModule
from titans.dataloader.imagenet import DaliDataloaderWithRandAug, DaliDataloader


class DaliImagenetDataModule(LightningDataModule):
    def __init__(self,
                 root: str,
                 batch_size=128,
                 num_threads=4,
                 resize=256,
                 crop=224,
                 prefetch=2,
                 cuda=True,
                 gpu_aug=True,
                 mixup_alpha=0.0,
                 randaug_magnitude=10,
                 randaug_num_layers=0) -> None:
        super().__init__()
        self.train_files = sorted(glob.glob(os.path.join(root, 'train/*')))
        self.train_idx_files = sorted(glob.glob(os.path.join(root, 'idx_files/train/*')))
        self.valid_files = sorted(glob.glob(os.path.join(root, 'validation/*')))
        self.valid_idx_files = sorted(glob.glob(os.path.join(root, 'idx_files/validation/*')))
        self.dataloader_args = dict(
            batch_size=batch_size,
            num_threads=num_threads,
            resize=resize,
            crop=crop,
            prefetch=prefetch,
            cuda=cuda,
            gpu_aug=gpu_aug
        )
        self.randaug_args = dict(
            mixup_alpha=mixup_alpha,
            randaug_magnitude=randaug_magnitude,
            randaug_num_layers=randaug_num_layers,
        )

    @property
    def use_randaug(self):
        return self.randaug_args['randaug_magnitude'] > 0 or self.randaug_args['mixup_alpha'] > 0.0

    def train_dataloader(self):
        if self.use_randaug:
            return DaliDataloaderWithRandAug(self.train_files,
                                             self.train_idx_files,
                                             shard_id=self.trainer.global_rank,
                                             num_shards=self.trainer.world_size,
                                             training=True,
                                             **self.dataloader_args,
                                             **self.randaug_args)
        return DaliDataloader(self.train_files,
                              self.train_idx_files,
                              shard_id=self.trainer.global_rank,
                              num_shards=self.trainer.world_size,
                              training=True,
                              **self.dataloader_args)

    def val_dataloader(self):
        if self.use_randaug:
            return DaliDataloaderWithRandAug(self.valid_files,
                                             self.valid_idx_files,
                                             shard_id=self.trainer.global_rank,
                                             num_shards=self.trainer.world_size,
                                             training=False,
                                             **self.dataloader_args,
                                             **self.randaug_args)
        return DaliDataloader(self.valid_files,
                              self.valid_idx_files,
                              shard_id=self.trainer.global_rank,
                              num_shards=self.trainer.world_size,
                              training=False,
                              **self.dataloader_args)
