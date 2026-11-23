# polystrata:p19-mmq - polystrata:next-c868922e with strata-poly built against the vendored llama.cpp tree
# (third_party/strata-llama) and -DSTRATA_POLY_MMQ=ON: llama.cpp MMQ prompt products and the FATTN prompt kernel.
FROM polystrata:next-c868922e
WORKDIR /opt/strata/terminal
RUN .venv/bin/cmake -G Ninja -DCMAKE_MAKE_PROGRAM=/opt/strata/terminal/.venv/bin/ninja -S /opt/strata/engine -B /tmp/b \
      -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_PORTABLE=OFF \
      -DSTRATA_GGML_DIR=/opt/strata/engine/third_party/strata-llama -DSTRATA_POLY_MMQ=ON \
    && .venv/bin/cmake --build /tmp/b --target strata-poly -j 6 \
    && cp /tmp/b/strata-poly engine-poly/strata-poly && rm -rf /tmp/b
