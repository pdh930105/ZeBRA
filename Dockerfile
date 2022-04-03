ARG BASE_VERSION=21.11
FROM nvcr.io/nvidia/pytorch:${BASE_VERSION}-py3 as base
FROM base as zebra
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y
#RUN apt-get upgrade -y

RUN git clone https://github.com/pdh930105/zebra.git
RUN pip install tqdm

#ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]
