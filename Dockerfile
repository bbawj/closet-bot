FROM python:3.11-slim-bookworm AS model-builder
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/
ADD ./models/ /app
ADD https://github.com/bbawj/closet-bot/releases/download/v0.1/finetuned_clip.pt /app
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN uv sync --locked
RUN uv run gen.py

FROM rust:1.84 AS builder

WORKDIR /app
COPY . .
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libssl3 ca-certificates

COPY --from=model-builder /app/clip-image-vit-32.onnx .
COPY --from=model-builder /app/clip-text-vit-32.onnx .
COPY --from=model-builder /app/yolo11n.onnx .
COPY --from=builder /app/target/release/closet-bot .
COPY --from=builder /app/target/release/libonnxruntime.so.1.16.0 .
COPY ./tokenizer.json .

ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

CMD ["./closet-bot"]
