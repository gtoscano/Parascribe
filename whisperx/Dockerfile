# WhisperX API image — built for Blackwell (RTX PRO 6000 / sm_120).
# Base provides torch 2.7 + CUDA 12.8 + cuDNN 9, which CTranslate2/faster-whisper
# need for Blackwell. Do NOT let pip downgrade torch (whisperx pulls it in).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

# ffmpeg is required by whisper/whisperx to decode audio; libsndfile for pyannote.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# Pin the base image's Blackwell torch so nothing downgrades it, then install in
# two stages — one big unconstrained install of pyannote's tree makes pip's
# resolver bail with "resolution-too-deep", so let whisperx resolve its own
# pinned deps first, then add the web server.
RUN printf 'torch==2.7.1\ntorchaudio==2.7.1\nnumpy<2.2\n' > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt -r requirements.txt \
    && pip install -c /tmp/constraints.txt \
        "fastapi==0.115.*" "uvicorn[standard]==0.34.*" python-multipart \
    # whisperx pins ctranslate2<4.5.0, but 4.4.0 links cuDNN 8 while this image
    # (and Blackwell) needs the cuDNN 9 build. 4.8.0 works with faster-whisper.
    && pip install --no-deps "ctranslate2==4.8.0" \
    # whisperx pulls huggingface_hub 1.x / transformers 5.x, but pyannote.audio
    # 3.4 still calls hf_hub_download(use_auth_token=...), removed in hf_hub 1.0.
    # Pin both back to the last compatible line.
    && pip install -c /tmp/constraints.txt \
        "huggingface_hub==0.34.4" "transformers==4.48.3"

COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
