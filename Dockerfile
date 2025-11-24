FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.6 8.7 9.0+PTX"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    SAM2_BUILD_ALLOW_ERRORS=0 \
    SAM2_BUILD_CUDA=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils \
    python3-pip git build-essential ninja-build ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip setuptools wheel build

RUN python3 -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY . .

# Install dependencies (use no build isolation to avoid conflicts with torch versions when building MMCV)
RUN python3 -m pip install --no-build-isolation -v .