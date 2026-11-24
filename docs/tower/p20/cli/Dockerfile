# polystrata:p20-x4: next-aadae31e with STRATA_CPU_NO_X4=1 (one weight row a call in the CPU experts) as a switch
FROM polystrata:next-aadae31e
COPY cpu_experts.cpp /opt/strata/engine/src/poly/cpu_experts.cpp
WORKDIR /opt/strata/terminal
RUN .venv/bin/python -c "import setup; setup.build_engine_poly({\"arch\": 86, \"archs\": [86]}, True, setup.get_llama_cpp(), \"cpu\", 13)" \
    && rm -rf build-poly build-vision-poly
