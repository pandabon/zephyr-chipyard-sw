.. zephyr:code-sample:: exeggutorch
   :name: ExecuTorch inference on RISC-V (exeggutorch)

   Test the ExecuTorch lowering and inference flow on RISC-V using XNNPack backend.
   Supports multiple models (MobileNetV2, LeNet, TinyMLP, SimpleTransformer, etc.)
   as stepping stones toward VLA deployment on target hardware.

   Python Dependencies:
   - Executorch
   - flatc

   To use, first run `./generate_pte.sh` to generate the model_pte.c file.

   To modify the example model, please modify `gen_pte.py`.

   To switch on/off the RISC-V Vector micro-kernels in XNNPack, set the CMake option ``XNNPACK_ENABLE_RISCV_VECTOR`` to ``ON`` or ``OFF``.

   Some toolchain may need a patch to C header files to support XNNPack. The gcc14.patch is targeted for the RISC-V GCC14.

