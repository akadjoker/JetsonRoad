FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.6 python3.6-dev python3-pip python3-opencv \
    build-essential \
    libjpeg-dev zlib1g-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    wget curl git unzip \
    cmake \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    pkg-config \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.6 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir scikit-build ninja
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

CMD ["bash"]


