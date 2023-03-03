# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/csrc/layer_norm/setup.py
# and https://github.com/NVIDIA/apex/blob/master/setup.py

import os
import subprocess
from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                        CUDAExtension)  
      
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. PyTorch is required to install mosaic_cuda_lib. Please ensure you have installed the main llm folder first with `pip install -e .[llm]`.")


if not 'cu' in torch.__version__:
    raise RuntimeError("CUDA version of PyTorch required to install optimizations.")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + '/bin/nvcc', '-V'],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index('release') + 1
    release = output[release_idx].split('.')
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f'{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  '
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc.")


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(
        CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ['--threads', '4']
    return nvcc_extra_args


TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(
        os.path.join(torch_dir, 'include', 'ATen', 'CUDAGeneratorImpl.h')):
    generator_flag = ['-DOLD_GENERATOR_PATH']

raise_if_cuda_home_none('--xentropy')
# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
cc_flag.append('-gencode')
cc_flag.append('arch=compute_70,code=sm_70')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_80,code=sm_80')

ext_modules.append(
    CUDAExtension(
        name='xentropy_cuda_lib',
        sources=['xentropy/interface.cpp', 'xentropy/xentropy_kernel.cu'],
        extra_compile_args={
            'cxx': ['-O3'] + generator_flag,
            'nvcc': append_nvcc_threads(['-O3'] + generator_flag + cc_flag),
        },
        include_dirs=[os.path.join(this_dir, 'xentropy')],
    ))

ext_modules.append(
    CUDAExtension(
        name='fused_dense_lib',
        sources=['fused_dense_lib/fused_dense.cpp', 'fused_dense_lib/fused_dense_cuda.cu'],
        extra_compile_args={
            'cxx': ['-O3',],
            'nvcc': append_nvcc_threads(['-O3'])
        },
        include_dirs=[os.path.join(this_dir, 'fused_dense_lib')],
    ))

ext_modules.append(
    CUDAExtension(
        name='dropout_layer_norm',
        sources=[
            'layer_norm/ln_api.cpp',
            'layer_norm/ln_fwd_256.cu',
            'layer_norm/ln_bwd_256.cu',
            'layer_norm/ln_fwd_512.cu',
            'layer_norm/ln_bwd_512.cu',
            'layer_norm/ln_fwd_768.cu',
            'layer_norm/ln_bwd_768.cu',
            'layer_norm/ln_fwd_1024.cu',
            'layer_norm/ln_bwd_1024.cu',
            'layer_norm/ln_fwd_1280.cu',
            'layer_norm/ln_bwd_1280.cu',
            'layer_norm/ln_fwd_1536.cu',
            'layer_norm/ln_bwd_1536.cu',
            'layer_norm/ln_fwd_2048.cu',
            'layer_norm/ln_bwd_2048.cu',
            'layer_norm/ln_fwd_2560.cu',
            'layer_norm/ln_bwd_2560.cu',
            'layer_norm/ln_fwd_3072.cu',
            'layer_norm/ln_bwd_3072.cu',
            'layer_norm/ln_fwd_4096.cu',
            'layer_norm/ln_bwd_4096.cu',
            'layer_norm/ln_fwd_5120.cu',
            'layer_norm/ln_bwd_5120.cu',
            'layer_norm/ln_fwd_6144.cu',
            'layer_norm/ln_bwd_6144.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'] + generator_flag,
            'nvcc':
                append_nvcc_threads([
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                ] + generator_flag + cc_flag),
        },
        include_dirs=[os.path.join(this_dir, 'layer_norm')],
    ))

setup(
    name='mosaic_cuda_lib',
    version='0.1',
    description='CUDA optimizations for common layers',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
