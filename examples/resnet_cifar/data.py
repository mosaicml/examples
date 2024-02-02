# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer
to the `CIFAR dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more
details.
"""
import numpy as np
import torch
import List, Tuple, Image, Union
import warnings 

from typing import Any, Callable, Optional

from composer.core import DataSpec
from composer.utils import dist
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

__all__ = ['StreamingCIFAR', 'build_cifar10_dataspec']


def pil_image_collate(
        batch: List[Tuple[Image.Image, Union[Image.Image, np.ndarray]]],
        memory_format: torch.memory_format = torch.contiguous_format) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constructs a length 2 tuple of torch.Tensors from datasets that yield samples of type
    :class:`PIL.Image.Image`.
    This function can be used as the ``collate_fn`` argument of a :class:`torch.utils.data.DataLoader`.
    Args:
        batch (List[Tuple[Image.Image, Union[Image.Image, np.ndarray]]]): List of (image, target) tuples
            that will be aggregated and converted into a single (:class:`~torch.Tensor`, :class:`~torch.Tensor`)
            tuple.
        memory_format (torch.memory_format): The memory format for the input and target tensors.
    Returns:
        (torch.Tensor, torch.Tensor): Tuple of (image tensor, target tensor)
            The image tensor will be four-dimensional (NCHW or NHWC, depending on the ``memory_format``).
    """
    warnings.warn(DeprecationWarning('pil_image_collate is deprecated and will be removed in v0.18'))
    imgs = [sample[0] for sample in batch]
    w, h = imgs[0].size
    image_tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)

    # Convert targets to torch tensor
    targets = [sample[1] for sample in batch]
    if isinstance(targets[0], Image.Image):
        target_dims = (len(targets), targets[0].size[1], targets[0].size[0])
    else:
        target_dims = (len(targets),)
    target_tensor = torch.zeros(target_dims, dtype=torch.int64).contiguous(memory_format=memory_format)

    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2).copy()
        if nump_array.shape[0] != 3:
            assert nump_array.shape[0] == 1, 'unexpected shape'
            nump_array = np.resize(nump_array, (3, h, w))
        assert image_tensor.shape[1:] == nump_array.shape, 'shape mismatch'

        image_tensor[i] += torch.from_numpy(nump_array)
        target_tensor[i] += torch.from_numpy(np.array(targets[i], dtype=np.int64))

    return image_tensor, target_tensor

class NormalizationFn:
    """Normalizes input data and removes the background class from target data if desired.
    An instance of this class can be used as the ``device_transforms`` argument
    when constructing a :class:`~composer.core.data_spec.DataSpec`. When used here,
    the data will normalized after it has been loaded onto the device (i.e., GPU).
    Args:
        mean (Tuple[float, float, float]): The mean pixel value for each channel (RGB) for
            the dataset.
        std (Tuple[float, float, float]): The standard deviation pixel value for each
            channel (RGB) for the dataset.
        ignore_background (bool): If ``True``, ignore the background class in the training
            loss. Only used in semantic segmentation. Default: ``False``.
    """

    def __init__(self,
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float],
                 ignore_background: bool = False):
        warnings.warn(DeprecationWarning('NormalizationFn is deprecated and will be removed in v0.18'))
        self.mean = mean
        self.std = std
        self.ignore_background = ignore_background

    def __call__(self, batch):
        xs, ys = batch
        assert isinstance(xs, torch.Tensor)
        assert isinstance(ys, torch.Tensor)
        device = xs.device

        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, device=device)
            self.mean = self.mean.view(1, 3, 1, 1)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(self.std, device=device)
            self.std = self.std.view(1, 3, 1, 1)

        xs = xs.float()
        xs = xs.sub_(self.mean).div_(self.std)
        if self.ignore_background:
            ys = ys.sub_(1)
        return xs, ys

# Scale by 255 since the collate `pil_image_collate` results in images in range 0-255
# If using ToTensor() and the default collate, remove the scaling by 255
CIFAR10_CHANNEL_MEAN = 0.4914 * 255, 0.4822 * 255, 0.4465 * 255
CIFAR10_CHANNEL_STD = 0.247 * 255, 0.243 * 255, 0.261 * 255


class StreamingCIFAR(StreamingDataset, VisionDataset):
    """CIFAR streaming dataset based on PyTorch's VisionDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset
            is stored.
        local (str): Local filesystem directory where dataset is cached during
            operation.
        split (str): The dataset split to use, either 'train' or 'test'.
        shuffle (bool): Whether to iterate over the samples in randomized order.
        transform (callable, optional): A function/transform that takes in an
            image and returns a transformed version. Default: ``None``.
        batch_size (int, optional): The batch size that will be used on each
            device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 transform: Optional[Callable] = None,
                 batch_size: Optional[int] = None) -> None:

        if split not in ['train', 'test']:
            raise ValueError(
                f"split='{split}', but must be one of ['train', 'test'].")

        self.transform = transform

        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         batch_size=batch_size)

    def __getitem__(self, idx: int) -> Any:
        sample = super().__getitem__(idx)
        image = sample['x']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = sample['y']
        return image, target


def build_cifar10_dataspec(
    data_path: str,
    is_streaming: bool,
    batch_size: int,
    local: Optional[str] = None,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> DataSpec:
    """Builds a CIFAR-10 dataloader with default transforms.

    Args:
        data_path (str): Path to the data; can be a local path or s3 path
        is_streaming (bool): Whether the data is stored locally or remotely
            (e.g. in an S3 bucket).
        batch_size (int): Batch size per device.
        local (str, optional): If using streaming, local filesystem directory
            where data is cached during operation. Default: ``None``.
        is_train (bool): Whether to load the training data or validation data.
            Default: ``True``.
        download (bool, optional): Whether to download the data locally, if
            needed. Default: ``True``.
        drop_last (bool): Drop remainder samples. Default: ``True``.
        shuffle (bool): Shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Any): Additional settings for the dataloader
            (e.g. num_workers, etc.).
    """
    if is_streaming and not local:
        raise ValueError(
            '`local` argument must be specified if using a streaming dataset.')

    split = 'train' if is_train else 'test'
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform = None

    if is_streaming:
        assert local is not None  # for type checkings
        dataset = StreamingCIFAR(
            remote=data_path,
            local=local,
            split=split,
            shuffle=shuffle,
            transform=transform,  # type: ignore
            batch_size=batch_size)
        sampler = None
    else:
        with dist.run_local_rank_zero_first():
            dataset = datasets.CIFAR10(root=data_path,
                                       train=is_train,
                                       download=dist.get_local_rank() == 0 and
                                       download,
                                       transform=transform)
        sampler = dist.get_sampler(dataset,
                                   drop_last=drop_last,
                                   shuffle=shuffle)

    device_transform_fn = NormalizationFn(mean=CIFAR10_CHANNEL_MEAN,
                                          std=CIFAR10_CHANNEL_STD)

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )
