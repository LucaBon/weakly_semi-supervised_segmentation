FROM nvcr.io/nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
    ca-certificates \
    python3-pip \
    python3.6-dev \
    python3.6 \
    curl \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    wget \
 && curl https://bootstrap.pypa.io/get-pip.py | python3.6 \
 # Cleaning after installations
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip==20.2.2 setuptools==49.6.0

RUN mkdir /app
WORKDIR /app

ADD requirements.txt .

RUN pip3 install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt