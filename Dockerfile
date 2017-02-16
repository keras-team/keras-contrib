FROM nvidia/cuda:8.0-cudnn5-devel

# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin
ENV HOME /root
ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL en_US.UTF-8
ENV TERM xterm

RUN locale-gen en_US.UTF-8

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git

# Use bash when building the image
RUN mv /bin/sh /bin/sh.orig && ln -s /bin/bash /bin/sh

# Install miniconda

ARG PYTHON_VERSION
# Invalidate docker cache
ADD README.md .
RUN echo "Python version: $PYTHON_VERSION"
RUN if [[ "$PYTHON_VERSION" == "2.7" ]]; then \
      wget --quiet https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O ~/miniconda.sh; \
    else \
      wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh; \
    fi
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN hash -r
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda

# Useful for debugging any issues with conda
RUN conda info -a

RUN conda install python=$PYTHON_VERSION numpy scipy matplotlib pandas pytest h5py
RUN pip install pytest-cov python-coveralls pytest-xdist coverage==3.7.1 #we need this version of coverage for coveralls.io to work
RUN pip install pep8 pytest-pep8
RUN pip install git+git://github.com/Theano/Theano.git
RUN python -c "import theano"

# install PIL for preprocessing tests
RUN if [[ "$PYTHON_VERSION" == "2.7" ]]; then \
      conda install pil; \
    elif [[ "$PYTHON_VERSION" == "3.5" ]]; then \
      conda install Pillow; \
    fi

# install TensorFlow
RUN if [[ "$PYTHON_VERSION" == "2.7" ]]; then \
      pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl; \
    elif [[ "$PYTHON_VERSION" == "3.5" ]]; then \
      pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl; \
    fi

RUN pip install git+git://github.com/fchollet/keras.git
RUN python -c "import keras.backend"
RUN mkdir -p ~/.keras/datasets

WORKDIR /src

ADD . /src

RUN pip install -e .


# Restore default shell
RUN rm /bin/sh && mv /bin/sh.orig /bin/sh
