# Imagem de GPU com PyTorch e CUDA. O torch do pip padrão é CPU e quebra a fala.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
        "numpy>=1.24.0,<2" \
        librosa==0.11.0 \
        s3tokenizer \
        tokenizers \
        transformers==5.2.0 \
        diffusers==0.29.0 \
        "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@ff1c8ac55a976971245cdd53c18d6131ca00d993" \
        conformer==0.3.2 \
        safetensors==0.5.3 \
        pyloudnorm \
        omegaconf \
        huggingface_hub \
        hf-transfer \
        soundfile \
        scipy \
        einops \
        runpod

# Sem as dependências: o pacote puxaria o torch de CPU por cima do CUDA.
# Commit fixo: o handler usa prepare_conditionals e model.conds desta versão.
RUN pip install --no-cache-dir --no-deps "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"

ENV HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

COPY handler.py resolve_reference.py voice_cache.py encode_wav.py /app/

# Baixa o pt-BR e confere se os pesos encaixam. Se não encaixarem, a imagem não fica pronta.
RUN python -c "import handler; handler.download_weights(); handler.load_brazilian(); print('pt-BR ok')"

CMD ["python", "-u", "/app/handler.py"]
