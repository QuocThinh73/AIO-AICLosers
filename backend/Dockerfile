FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_HOME=/hf-cache \
    TRANSFORMERS_CACHE=/hf-cache/transformers \
    HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
    TORCH_HOME=/hf-cache/torch

WORKDIR /backend

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /hf-cache/transformers /hf-cache/hub /hf-cache/torch

VOLUME ["/hf-cache"]

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
