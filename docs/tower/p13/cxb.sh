#!/bin/sh
# Pass 13: builds cxb (cxb-t with the tower branch's cpu_experts.cpp, cxb-b with PR 36's) in a throwaway container
# of IMAGE, against ggml-cpu from the image's pinned llama.cpp, built native as the engine's is.
# On Tower: sh cxb.sh build IMAGE, then sh cxb.sh run IMAGE ARGS... (the engine container stopped).
set -e
D=/mnt/user/appdata/polystrata/p13
GG=/gguf/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
IMG=$2
case "$1" in
build)
  docker run --rm -v $D:/w -v /mnt/cache/llm:/gguf --entrypoint sh "$IMG" -c '
    set -e; cd /opt/strata; export PATH=/opt/strata/.venv/bin:$PATH
    PYTHONPATH=third_party/llama.cpp/gguf-py .venv/bin/python /w/exptab.py '$GG' > /w/tab.txt
    cmake -S third_party/llama.cpp -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF -B /w/ggml-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DGGML_BACKEND_DL=OFF -DGGML_BUILD_TESTS=OFF \
      -DGGML_BUILD_EXAMPLES=OFF > /w/ggml-cmake.log
    cmake --build /w/ggml-build --target ggml-cpu ggml-base > /w/ggml-build.log
    INC="-I/opt/strata/src/poly -Ithird_party/llama.cpp/ggml/include"
    LIBS="$(find /w/ggml-build -name libggml-cpu.a) $(find /w/ggml-build -name libggml-base.a) -lpthread -lm -ldl"
    for v in t b; do
      src=/opt/strata/src/poly/cpu_experts.cpp; [ $v = b ] && src=/w/pr36/cpu_experts.cpp
      g++ -O3 -DNDEBUG -std=c++20 $INC /w/cxb.cpp $src $LIBS -o /w/cxb-$v
    done
    cat /w/tab.txt | head -3; ls -la /w/cxb-t /w/cxb-b'
  ;;
run)
  shift 2
  docker run --rm -v $D:/w -v /mnt/cache/llm:/gguf --entrypoint sh "$IMG" -c "cd /w && $*"
  ;;
esac
