
FROM ubuntu:bionic
LABEL maintainer="Shyam Sudhakaran <shyamsnair97@gmail.com>"                                           

ENV                                    \
  DEBIAN_FRONTEND=noninteractive                                       \
  LANG=C.UTF-8                                                         \
  LC_ALL=C.UTF-8                                                       \
  PATH=/opt/conda/bin:$PATH                                            \
TZ=America/Los_Angeles     

# Install system packages.
RUN apt-get update       \
  && apt-get install -y  \
    build-essential      \
    bzip2                \
    ca-certificates      \
    curl                 \
    git                  \
    tzdata               \
    wget                 \
    gcc                  \
    libgl1-mesa-glx      \
  && apt-get clean       \
  && apt-get autoremove  \
  && rm -rf /var/lib/apt/lists/*

RUN apt update -y 
RUN apt install openjdk-8-jre-headless -y 

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN conda create -n env -y python=3.8.0
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

ENV CODE_ROOT /code/3d-artefacts-nca

RUN mkdir -p $CODE_ROOT
COPY . $CODE_ROOT
WORKDIR $CODE_ROOT

RUN python setup.py install