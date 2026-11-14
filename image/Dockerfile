# Clean distribution: copy selected dependencies, never inherit the runtime's
# previous application layers, home directories or experiment history.
ARG RUNTIME_IMAGE=ghcr.io/neaoo/inferboost:0.5.0-cu130-r11
FROM ${RUNTIME_IMAGE} AS tested-runtime
FROM ubuntu:22.04@sha256:8a2321a92b438238f93b4346fff61bc4259c99fdd4d9b52ee86f153fee0c344f
USER root
ARG PRODUCT_VERSION=0.5.0-r11
ARG SOURCE_REVISION=unrecorded
RUN apt-get -o Acquire::Retries=2 update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates build-essential binutils libnuma1 libgomp1 libssl3 libffi8 \
       libsqlite3-0 libbz2-1.0 liblzma5 zlib1g libexpat1 libreadline8 libncursesw6 tzdata curl git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=tested-runtime /usr/local/bin/uv /usr/local/bin/uv
COPY --from=tested-runtime /opt/python /opt/python
COPY --from=tested-runtime /opt/venv /opt/venv
COPY --from=tested-runtime /opt/quantization /opt/quantization
COPY --from=tested-runtime /opt/inferboost-ui /opt/inferboost-ui
COPY --from=tested-runtime /usr/local/cuda-13.0 /usr/local/cuda-13.0
RUN ln -s /usr/local/cuda-13.0 /usr/local/cuda && ln -s /opt/venv /opt/inference
ENV PATH=/opt/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CUDA_HOME=/usr/local/cuda-13.0 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1 ACCEL60_DRAFT_CONFIG_FIX=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    INFERBOOST_DEFAULT_WORKSPACE=/workspace/inferboost-v050 PIPELINE_WORKSPACE=/workspace/inferboost-v050 \
    PIPELINE_PYTHON=/opt/inferboost-ui/bin/python
COPY pipeline /opt/pipeline/pipeline
COPY scripts /opt/pipeline/scripts
COPY configs/*.example.json /opt/pipeline/configs/
COPY data/public100.json data/quality100.json /opt/pipeline/data/
COPY quickstart inferboost accelerate README.md README.en.md CONTRIBUTING.md CHANGELOG.md LICENSE* THIRD_PARTY_NOTICES.md /opt/pipeline/
COPY docs /opt/pipeline/docs
COPY image /opt/pipeline/image
COPY requirements /opt/pipeline/requirements
COPY references/*.json /opt/pipeline/references/
COPY references/model-configs/qwen3.*.json /opt/pipeline/references/model-configs/
COPY image/container-entry.sh /usr/local/bin/inferboost-container
COPY image/bootstrap.sh /usr/local/bin/quickstart
COPY image/welcome.sh /etc/profile.d/inferboost.sh
RUN chmod +x /usr/local/bin/inferboost-container /usr/local/bin/quickstart /opt/pipeline/quickstart /opt/pipeline/inferboost /opt/pipeline/accelerate \
    && chmod -R a+rX /opt/pipeline \
    && ln -sf /usr/local/bin/quickstart /usr/local/bin/inferboost \
    && printf '\n. /etc/profile.d/inferboost.sh\n' >> /etc/bash.bashrc \
    && /opt/venv/bin/python /opt/pipeline/image/verify_image.py \
    && /opt/quantization/bin/python /opt/pipeline/image/verify_image.py --quantization \
    && /opt/venv/bin/python /opt/pipeline/image/workspace.py manifest /opt/pipeline
WORKDIR /workspace
EXPOSE 9001
LABEL org.opencontainers.image.title="InferBoost" \
      org.opencontainers.image.description="Interactive LLM quantization, measured inference tuning and portable deployment" \
      org.opencontainers.image.version="${PRODUCT_VERSION}" \
      org.opencontainers.image.source="https://github.com/NeAoo/InferBoost" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.inferboost.variant="cli"
ENTRYPOINT ["/usr/local/bin/inferboost-container"]
CMD []
