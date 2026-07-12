# GLM-5.2-NVFP4 serving image for 8x RTX PRO 6000 Blackwell (SM120, PCIe).
# Base: official vLLM v0.25.0 release image, then apply fix.patch (Python-only
# upstream backports + SM120 sparse-MLA DCP support) to the installed package.
FROM vllm/vllm-openai:v0.25.0

# fix.patch is git-format; git handles mbox-style concatenated patches cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends git patch \
    && rm -rf /var/lib/apt/lists/*

COPY fix.patch /opt/fix.patch

# Apply to the installed vllm package (tests/ hunks are skipped, not shipped
# in the image). Fails the build loudly if any hunk does not apply.
RUN set -eux; \
    SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))"); \
    cd "$SITE"; \
    git apply --verbose -p1 --include='vllm/*' /opt/fix.patch

# FlashInfer 0.6.14 with the prebuilt cu130 AOT kernel cache. The jit-cache
# wheel is THE critical piece on CUDA-13/SM120: it avoids JIT compilation,
# which fails on this toolchain. FLASHINFER_DISABLE_VERSION_CHECK is needed
# because the cubin package lags at 0.6.13.
RUN pip install --no-cache-dir 'flashinfer-python==0.6.14' \
    && pip install --no-cache-dir 'flashinfer-jit-cache==0.6.14' \
       --index-url https://flashinfer.ai/whl/cu130

# The FlashInfer cu130 prebuilt kernels dlopen libnvrtc.so.13 / libcudart.so.13,
# which live in the pip "nvidia" packages, not in the image's CUDA dirs.
# Keep the base image paths appended.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

# Sanity check at build time.
RUN python3 -c "import vllm, flashinfer, torch; print('vllm', vllm.__version__, '| flashinfer', flashinfer.__version__, '| torch', torch.__version__)"
