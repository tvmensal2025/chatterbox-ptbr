"""Worker da Runpod: Chatterbox em português do Brasil, com clonagem por link."""
import base64
import os
from pathlib import Path

import runpod

from encode_wav import wav_bytes
from resolve_reference import materialize_reference
from voice_cache import VoiceCache

BASE_REPO = "ResembleAI/chatterbox"
BRAZIL_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"
DEFAULT_VOICE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav"
DEFAULT_VOICE_PATH = "/tmp/pt-br-default.wav"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
WARM_UP_TEXT = "Olá, tudo bem?"

MODEL = None
DEFAULT_VOICE = None


def download_weights() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.getenv("HF_TOKEN") or None
    snapshot_download(
        repo_id=BASE_REPO,
        allow_patterns=["ve.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"],
        token=token,
    )
    hf_hub_download(BRAZIL_REPO, "t3_pt_br.safetensors", token=token)
    hf_hub_download(BRAZIL_REPO, "s3gen_v3.pt", token=token)


def _tensor_state(loaded):
    if not isinstance(loaded, dict):
        return loaded
    for key in ("state_dict", "model"):
        value = loaded.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            return value[0]
    return loaded


def load_brazilian():
    import torch
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file

    token = os.getenv("HF_TOKEN") or None
    base = Path(snapshot_download(
        repo_id=BASE_REPO,
        allow_patterns=["ve.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"],
        token=token,
    ))
    t3_path = hf_hub_download(BRAZIL_REPO, "t3_pt_br.safetensors", token=token)
    s3_path = hf_hub_download(BRAZIL_REPO, "s3gen_v3.pt", token=token)

    voice = VoiceEncoder()
    voice.load_state_dict(torch.load(base / "ve.pt", map_location="cpu", weights_only=True))
    voice.to(DEVICE).eval()

    t3 = T3(T3Config.multilingual())
    t3_state = _tensor_state(load_file(t3_path))
    t3.load_state_dict(t3_state)
    t3.to(DEVICE).eval()

    s3gen = S3Gen()
    s3_state = _tensor_state(torch.load(s3_path, map_location="cpu", weights_only=True))
    report = s3gen.load_state_dict(s3_state, strict=False)
    ignored = set(getattr(s3gen, "ignore_state_dict_missing", ()))
    missing = [key for key in report.missing_keys if key not in ignored]
    if missing:
        raise RuntimeError("Pesos do decodificador incompletos: " + ", ".join(missing[:12]))
    s3gen.to(DEVICE).eval()

    tokenizer = MTLTokenizer(str(base / "grapheme_mtl_merged_expanded_v1.json"))
    return ChatterboxMultilingualTTS(t3, s3gen, voice, tokenizer, DEVICE)


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_brazilian()
    return MODEL


def ensure_default_voice() -> str:
    global DEFAULT_VOICE
    if DEFAULT_VOICE and os.path.isfile(DEFAULT_VOICE) and os.path.getsize(DEFAULT_VOICE) > 44:
        return DEFAULT_VOICE
    materialize_reference({"reference_audio_url": DEFAULT_VOICE_URL}, DEFAULT_VOICE_PATH)
    DEFAULT_VOICE = DEFAULT_VOICE_PATH
    return DEFAULT_VOICE


VOICES = VoiceCache(ensure_default_voice)


def handler(job):
    job_input = job.get("input") or {}
    if job_input.get("health_check"):
        return {"status": "healthy", "language": "pt-BR", "model_loaded": MODEL is not None}

    text = str(job_input.get("text") or "").strip()
    if not text:
        return {"error": "No text provided"}

    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    temperature = float(job_input.get("temperature", 0.8))
    try:
        model = get_model()
        cloned = VOICES.use(model, job_input, exaggeration)
        wav = model.generate(
            text,
            language_id="pt",
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        return {
            "audio_base64": base64.b64encode(wav_bytes(wav.squeeze(0).numpy(), model.sr)).decode("utf-8"),
            "sample_rate": model.sr,
            "format": "wav",
            "language": "pt-BR",
            "cloned": cloned,
        }
    except Exception as error:
        return {"error": str(error)}


def prepare_worker() -> None:
    """Sem GPU o worker sai em vez de gerar na CPU, lenta demais para a ligação. A frase de aquecimento tira a lentidão da primeira fala."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Worker sem GPU CUDA. O Chatterbox pt-BR não atende na CPU.")
    print(f"GPU {torch.cuda.get_device_name(0)}")
    model = get_model()
    VOICES.use(model, {}, 0.5)
    model.generate(WARM_UP_TEXT, language_id="pt")
    print("Modelo pt-BR pronto")


if __name__ == "__main__":
    try:
        prepare_worker()
    except Exception as error:
        print(f"Falha ao preparar o modelo pt-BR: {error}")
    runpod.serverless.start({"handler": handler})
