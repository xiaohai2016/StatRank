FROM docker.artifactory-test.corp.linkedin.com/tensorflow/fairing-gpu:1.100

# version with CUDA 9.x available on Kubernetes pods
RUN conda update -n base conda && \
    conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch

# expose Python 3
ENV PATH="/opt/conda/bin/:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip install sklearn
