from torch.utils.cpp_extension import load

# 明确路径（请根据你的项目结构调整）
fused = load(
    name="fused",
    sources=[
        "op/fused_act.cpp",
        "op/fused_act_kernel.cu"
    ],
    extra_cuda_cflags=['-O2']
)

print("✅ 编译成功")