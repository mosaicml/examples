// Copied from https://github.com/HazyResearch/flash-attention/tree/eb33e587e95ec29a13c58f76dadca04b64122784/csrc/layer_norm

// Copyright (c) 2022, Tri Dao and Dan Fu. All rights reserved.
#include "ln_bwd_kernels.cuh"

// Create backward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_BWD_LAUNCHER( 2048, fp32, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp16, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp32, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp16, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp32, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp32, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, bf16, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp32, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, fp16, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_BWD_LAUNCHER( 2048, bf16, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
