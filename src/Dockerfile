FROM ubuntu:18.04

RUN apt update && apt install -y curl build-essential python3.7 python3.7-distutils vim

RUN curl -fsSL https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

CMD ["bash"]
