FROM nvidia/cuda:12.2.2-base-ubuntu22.04

WORKDIR /usr/src/app
COPY . .

ENV PYTHONUNBUFFERED=True
ENV IMAGEIO_FFMPEG_EXE=ffmpeg
ENV NEMO_HOME_DIR="/usr/src/app/"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
	apt install -y lsb-release build-essential ffmpeg gnupg software-properties-common curl libsndfile1 git python3.10 python3.10-venv python3-pip && \ 
	rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m venv /opt/nemo
# RUN source nemo/bin/activate
ENV PATH="/opt/nemo/bin:$PATH"
# RUN source /opt/nemo/bin/activate
RUN pip install --upgrade setuptools wheel
RUN pip install Cython packaging
RUN pip install nemo_toolkit['tts']
RUN pip install numpy==1.26.4 matplotlib==3.7.0

CMD ["/bin/sleep", "infinity"]