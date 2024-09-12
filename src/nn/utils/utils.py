import torch


def set_jit_flags():
    # Flags required to enable JIT fusion kernels.
    # See: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/fused_bias_dropout.py.
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
