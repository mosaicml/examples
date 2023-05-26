# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

try:
    from data import StreamingCIFAR, build_cifar10_dataspec
    from model import ResNetCIFAR, build_composer_resnet_cifar
except ImportError as e:
    raise ImportError(
        'Please make sure to pip install the requirements.txt in the CIFAR benchmark.'
    ) from e

__all__ = [
    'StreamingCIFAR',
    'build_cifar10_dataspec',
    'ResNetCIFAR',
    'build_composer_resnet_cifar',
]
