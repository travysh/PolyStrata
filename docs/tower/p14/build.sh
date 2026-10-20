#!/bin/sh
# Pass 14: builds cxb-<name> for each variant of cpu_experts_v.cpp, against p13's ggml-cpu build.
# On Tower: sh build.sh IMAGE name:"-DFLAG=1 ..." ...
D=/mnt/user/appdata/polystrata; IMG=$1; shift
for v in "$@"; do
  name=${v%%:*}; flags=${v#*:}
  docker run --rm -v $D/p14:/w -v $D/p13/ggml-build:/w/ggml-build --entrypoint sh $IMG -c "cd /opt/strata && \
    g++ -O3 -DNDEBUG -std=c++20 $flags -I/opt/strata/src/poly -Ithird_party/llama.cpp/ggml/include /w/cxb.cpp \
    /w/cpu_experts_v.cpp /w/ggml-build/ggml/src/libggml-cpu.a /w/ggml-build/ggml/src/libggml-base.a -lpthread -lm -ldl \
    -o /w/cxb-$name" && echo "built cxb-$name ($flags)"
done
