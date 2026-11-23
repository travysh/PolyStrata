# polystrata:next - PolyStrata 0.2.19-poly.1 (upstream main 50e42805) + the local tower-next branch (logit_bias, GBNF
# grammars, constrained response_format, assistant prefill), for Tower's RTX 3070 (sm_86). Upstream's Docker image
# has no strata-poly models, so this one builds strata-poly the way terminal/setup.py does and starts the server
# with the hand-kept config on /data. Build context: /mnt/user/appdata/polystrata/next (repo/ = git archive).
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv build-essential git ca-certificates libgomp1 libatomic1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
COPY repo/ /opt/strata/
COPY TOWER_COMMIT /opt/strata/TOWER_COMMIT
WORKDIR /opt/strata/terminal
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-build.txt
ENV PATH=/opt/strata/terminal/.venv/bin:$PATH
# the ported parts' own tests (CPU only), then the engine and the CPU image encoder for sm_86
RUN set -e; cd /opt/strata/engine; \
    g++ -std=c++20 -O1 -Iinclude -Isrc/poly src/poly/sampler_test.cpp src/poly/grammar.cpp -o /tmp/st && /tmp/st; \
    g++ -std=c++20 -O1 -Iinclude -Isrc/poly src/poly/grammar_test.cpp src/poly/grammar.cpp -o /tmp/gt && /tmp/gt > /dev/null; \
    rm -f /tmp/st /tmp/gt
RUN .venv/bin/python -c "import setup; setup.build_engine_poly({'arch': 86, 'archs': [86]}, True, setup.get_llama_cpp(), 'cpu', 13)" \
    && rm -rf build-poly build-vision-poly
EXPOSE 8080
CMD [".venv/bin/python", "serve/server.py", "--engine", "strata", "--config", "/data/config/strata-cybertiel-next.json", "--port", "8080", "--gpu", "0"]
