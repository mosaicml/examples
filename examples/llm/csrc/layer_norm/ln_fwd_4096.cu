// Copied from https://github.com/HazyResearch/flash-attention/tree/eb33e587e95ec29a13c58f76dadca04b64122784/csrc/layer_norm
#include "ln_fwd_kernels.cuh"

// Create forward launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, RYTPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_FWD_LAUNCHER( 4096, fp32, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp16, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp16, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, bf16, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp16, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, bf16, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
