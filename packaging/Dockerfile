# Build the Linux binary without a Linux machine.
#
#   docker build -f packaging/Dockerfile -t herta-build .
#   docker run --rm -v "$PWD/dist:/out" herta-build
#
# The result runs on any glibc distribution at least as new as the image
# (Ubuntu 22.04 → glibc 2.35), and bundles Qt WebEngine so the target needs
# no GTK/WebKit packages.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        libgl1 libegl1 libxkbcommon-x11-0 libdbus-1-3 libnss3 \
        libxcomposite1 libxdamage1 libxrandr2 libasound2 libxkbfile1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY requirements.txt requirements-linux.txt ./
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements-linux.txt

COPY . .
# build.sh looks for .venv-unix; point it at the prepared environment.
RUN ln -sfn /venv .venv-unix && \
    /venv/bin/pip install pyinstaller && \
    ./build.sh --qt --no-deps

CMD ["sh", "-c", "cp -v dist/HertaStation /out/ && ls -l /out/HertaStation"]
