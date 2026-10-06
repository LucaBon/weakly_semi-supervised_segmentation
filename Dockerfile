FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -yq --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen

ENTRYPOINT ["uv", "run"]
CMD ["python", "-m", "wsss.train", "--help"]
