## Overview

A collection of sample workloads and examples to be used with the Zephyr flow for Chipyard.

## Zephyr SDK Installation

The Zephyr SDK provides a pre-built toolchain with GNU and LLVM compilers, host tools, and CMake integration. This is an alternative to building the toolchain from source.

Clone and init submodules:
```bash
git clone git@github.com:pandabon/zephyr-chipyard-sw.git
cd zephyr-chipyard-sw
# ~15 minutes
git submodule update --init
```

If conda is not already installed:
```bash
# SKIP if conda already installed
source scripts/install_conda.sh
```

Install dependencies:
```bash
# ~25 minutes
bash scripts/install_submodules.sh
```

Install the Zephyr SDK (minimal SDK with RISC-V 64-bit support):
```bash
# ~25 minutes
bash scripts/install_toolchain_sdk.sh
```

Set environment variables:
```bash
source scripts/set_envvars_sdk.sh
```

Activate conda:
```bash
# or use your existing conda path
source tools/miniforge3/etc/profile.d/conda.sh
```

Activate the zephyr environment:
```bash
conda activate zephyr
```

To test an example with spike (spike requires a chipyard install):
```bash
west build -p -b spike_riscv64 samples/hello_world/
spike build/zephyr/zephyr.elf
```

<!-- ## Cross Compiler Installation

Clone and init submodules:
```bash
git clone git@github.com:pandabon/zephyr-chipyard-sw.git
cd zephyr-chipyard-sw
# ~15 minutes
git submodule update --init
```

If conda is not already installed:
```bash
# SKIP if conda already installed
source scripts/install_conda.sh
```

Install dependencies:
```bash
# ~25 minutes
bash scripts/install_submodules.sh
```

Install GCC15.1, patched for Zephyr support:
```bash
bash scripts/install_toolchain.sh
```

Set environment variables:
```bash
source scripts/set_envvars.sh
```

Activate conda:
```bash
# or use your existing conda path
source tools/miniforge3/etc/profile.d/conda.sh
```

Activate the zephyr environment:
```bash
conda activate zephyr
```

To test an example with spike (spike requires a chipyard install):
```bash
west build -p -b spike_riscv64 samples/hello_world/
spike build/zephyr/zephyr.elf
``` -->

## Executorch/Torch Installation

**Note:** The torch setup uses the `torch-bump-testing` branch of the main repository, which references specific commits for `zephyr_ws/zephyr`, `third-party/executorch`, and `third-party/XNNPACK` submodules.

After the main installation, checkout the torch-bump-testing branch:
```bash
git fetch origin
git checkout torch-bump-testing
bash scripts/install_torch.sh
```

To test an example using Executorch, inside `zephyr-chipyard-sw`:
```bash
# Generate executorch C headers
./samples/executorch/generate_pte.sh --model mnist_mlp

# Build with the RVV XNNPACK Runtime
west build -p -b spike_riscv64 ./samples/executorch/executor_runner/ -DXNNPACK_ENABLE_RISCV_VECTOR=ON -DXNNPACK_ENABLE_RISCV_GEMMINI=OFF

# Run using spike
spike -p4 --isa=rv64gcv_zicntr build/zephyr/zephyr.elf
```

#### `generate_pte.sh` Options

The `generate_pte.sh` script forwards all arguments to `gen_pte.py`. Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `mobilenet` | Model to export. Choices: `mobilenet`, `squeezenet`, `alexnet`, `mobilenetv3small`, `lenet`, `transformer`, `mnist_mlp` |
| `--precision PREC` | `fp32` | Model weight/activation dtype. Choices: `fp32`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2` |
| `--pte PATH` | `model.pte` | Output path for the PTE file |
| `--test` | off | Run a forward pass with random inputs and print results; skip export |
| `--lstats` | off | Print XNNPACK delegation summary and delegated graph after lowering |
| `--seed SEED` | none | Random seed for reproducible test inputs |
| `--deprecated-lower` | off | Use deprecated `to_edge()` + `to_backend()` flow instead of `to_edge_transform_and_lower()` |

Example:
```bash
./samples/executorch/generate_pte.sh --model lenet --precision fp16 --lstats
```

## Chipyard Installation

TODO

## Troubleshooting

If you encounter issues during installation or when activating the conda environment, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting information.
