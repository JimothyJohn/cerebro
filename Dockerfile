FROM python:3.9-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# https://voxel51.com/docs/fiftyone/environments/index.html#setting-up-a-cloud-instance
RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
    curl \
    build-essential \ 
    unzip \
    cmake \
    cmake-data \
    pkg-config \
    libsm6 \
    libxext6 \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    python3 \
    python-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg

COPY requirements.txt /
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app /app

# RUN useradd -U fiftyone 
# RUN chown fiftyone /app
# USER fiftyone

CMD ["python","/app"]
