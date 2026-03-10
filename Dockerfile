ARG TORCH_TAG=2.5.1-cuda12.4-cudnn9
FROM pytorch/pytorch:${TORCH_TAG}-devel

WORKDIR /app


# Arguments to build Docker Image using CUDA
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
ENV SAM2_BUILD_ALLOW_ERRORS=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONUNBUFFERED=1

# Grounded-SAM2 — use gcc-10 to match what it was tested with
ENV CC=gcc-10
ENV CXX=g++-10
ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA=1

RUN apt-get update && apt-get install -y \
    git build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    wget ffmpeg \
    nano vim ninja-build \
    gcc-10 g++-10 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir numpy wheel "setuptools>=62.3.0,<75.9" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ultralytics --no-deps


RUN TORCH_LIB_PATH=$(python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'lib')") && \
    echo "$TORCH_LIB_PATH" >> /etc/ld.so.conf.d/torch.conf && \
    ldconfig

COPY . /app


# Install segment_anything package in editable mode
RUN python -m pip install -e .

# Install grounding dino 
RUN python -m pip install --no-build-isolation -e grounding_dino

COPY serve /usr/bin/serve
RUN chmod +x /usr/bin/serve

EXPOSE 8080
ENTRYPOINT ["serve"]