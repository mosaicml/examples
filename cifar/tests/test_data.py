# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from ..data import build_cifar10_dataspec
from ..tests.utils import SynthClassificationDirectory

# TODO: streaming dataset and dataloader testing


@pytest.mark.parametrize('is_train', [True, False])
def test_dataloader_builder(is_train, batch_size=2):
    with SynthClassificationDirectory() as datadir:
        cifar_dataspec = build_cifar10_dataspec(data_path=datadir,
                                                is_streaming=True,
                                                local=datadir,
                                                batch_size=batch_size,
                                                is_train=is_train,
                                                download=False)

        print(len(cifar_dataspec.dataloader))
        assert len(cifar_dataspec.dataloader) == 8

        for i, batch in enumerate(cifar_dataspec.dataloader):
            # Check the image and label shapes
            assert batch[0].shape == torch.Size([batch_size, 3, 32, 32])
            assert batch[1].shape == torch.Size([batch_size])
