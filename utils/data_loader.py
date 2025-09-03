# utils/data_loader.py
import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def create_data_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 8,
    distributed: bool = False,
    input_size: int = 224,  # <-- 核心修改：使输入尺寸可配置
    augmentation: str = 'basic'
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    if dataset == 'imagenet':
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        # 训练集的数据增强
        train_transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4 if augmentation == 'strong' else None,
            auto_augment='rand-m9-mstd0.5-inc1' if augmentation == 'strong' else None,
            interpolation='bicubic',
            re_prob=0.25 if augmentation == 'strong' else 0.0,
            re_mode='pixel',
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        
        # 验证集的数据变换
        val_transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        val_dataset = datasets.ImageFolder(val_dir, val_transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader