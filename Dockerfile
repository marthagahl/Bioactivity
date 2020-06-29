FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN git clone https://github.com/nvidia/apex && \
    cd apex && \
    python setup.py install --user --cuda_ext --cpp_ext && \
    rm -rf /apex

RUN apt update && apt install -y rsync

RUN conda install -y -c conda-forge rdkit

ADD requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt

RUN rm -rf /tmp/*

CMD "/bin/bash"
