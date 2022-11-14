import os
from typing import Any, Callable, Optional, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset

from composer.core import DataSpec
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist
from streaming import Dataset


# Scale by 255 since the collate `pil_image_collate` results in images in range 0-255
# If using ToTensor() and the default collate, remove the scaling by 255
IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)

class StreamingImageNet(Dataset, VisionDataset):
    """ ImageNet streaming dataset based on PyTorch's Vision dataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to iterate over the samples in randomized order.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            Default: ``None``.
        prefetch (int, optional): Target number of samples remaining to prefetch while iterating.
            Default: ``100_000``.
        retry (int, optional): Number of download re-attempts before giving up. Defaults to ``2``.
        timeout (float, optional): Number of seconds to wait for a shard to download before raising
            an exception. Default: 120 seconds.
        batch_size (int, optional): Hint the batch size that will be used on each device's DataLoader.
            Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 transform: Optional[Callable] = None,
                 prefetch: Optional[int] = 100_000,
                 retry: int = 2,
                 timeout: float = 120,
                 batch_size: Optional[int] = None) -> None:

        if split not in ['train', 'val']:
            raise ValueError(
                f"split='{split}', but must be one of ['train', 'val'].")

        self.transform = transform

        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         prefetch=prefetch,
                         keep_zip=False,
                         retry=retry,
                         timeout=timeout,
                         hash=None,
                         batch_size=batch_size)

    def __getitem__(self, idx: int) -> Any:
        sample = super().__getitem__(idx)
        image = sample['x']
        if image.mode == "CMYK":
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = sample['y']
        return image, target


def build_streaming_imagenet_dataspec(
    remote: str,
    local: str,
    batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    **dataloader_kwargs,
) -> DataSpec:
    """Builds an ImageNet dataloader for local data.

    Args:
        datadir (str): path to location of dataset.
        batch_size (int): Batch size per device.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    split = 'train' if is_train else 'val'
    transform: List[torch.nn.Module] = []
    if resize_size > 0:
        transform.append(transforms.Resize(resize_size))

    # Add split specific transformations
    if is_train:
        transform += [
            transforms.RandomResizedCrop(crop_size,
                                         scale=(0.08, 1.0),
                                         ratio=(0.75, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip()
        ]
    else:
        transform.append(transforms.CenterCrop(crop_size))

    transform = transforms.Compose(transform)

    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                          std=IMAGENET_CHANNEL_STD)

    dataset = StreamingImageNet(remote=remote,
                                local=local,
                                split=split,
                                shuffle=shuffle,
                                transform=transform)

    # DataSpec allows for on-gpu transformations, slightly relieving dataloader bottleneck
    return DataSpec(
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )


def build_imagenet_dataspec(
    datadir: str,
    batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    **dataloader_kwargs,
) -> DataSpec:
    """Builds an ImageNet dataloader for local data.

    Args:
        datadir (str): path to location of dataset.
        batch_size (int): Batch size per device.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    split = 'train' if is_train else 'val'
    transformations: List[torch.nn.Module] = []
    if resize_size > 0:
        transformations.append(transforms.Resize(resize_size))

    # Add split specific transformations
    if is_train:
        transformations += [
            transforms.RandomResizedCrop(crop_size,
                                         scale=(0.08, 1.0),
                                         ratio=(0.75, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip()
        ]
    else:
        transformations.append(transforms.CenterCrop(crop_size))

    transformations = transforms.Compose(transformations)

    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                          std=IMAGENET_CHANNEL_STD)

    dataset = ImageFolder(os.path.join(datadir, split), transformations)
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    # DataSpec allows for on-gpu transformations, slightly relieving dataloader bottleneck
    return DataSpec(
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )

# Helpful to test if your dataloader is working locally
# Run `python data.py datadir` to test local
# Run `python data.py s3://my-bucket/my-dir/data /tmp/path/to/local` to test streaming dataset
if __name__ == '__main__':
    import sys
    from itertools import islice
    path = sys.argv[1]

    batch_size = 2
    if len(sys.argv) > 2:
        local = sys.argv[2]
        dataspec = build_streaming_imagenet_dataspec(remote=path, local=local, batch_size=batch_size)
    else:
        dataspec = build_imagenet_dataspec(datadir=path, batch_size=batch_size)

    print('Running 5 batchs of dataloader')
    for batch_ix, batch in enumerate(islice(dataspec.dataloader, 5)):
        print(batch_ix, ': Image batch shape:', batch[0].shape, 'Target batch shape:')
