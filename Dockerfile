FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.6 python3.6-dev python3-pip \
    build-essential \
    cmake \
    libjpeg-dev zlib1g-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    wget curl git unzip \
    && rm -rf /var/lib/apt/lists/*

# Symlinks
RUN ln -sf /usr/bin/python3.6 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copia requirements e instala tudo já no build
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

CMD ["bash"]

