"""Grava a voz de referência num arquivo local para o Chatterbox clonar."""
import base64
import os
import urllib.request

MAX_BYTES = 2_000_000


def materialize_reference(job_input: dict, dest: str) -> str | None:
    path = job_input.get("reference_audio_url") or job_input.get("audio_prompt_path")
    if isinstance(path, str) and path.startswith(("http://", "https://")):
        _download(path, dest)
        return dest
    if isinstance(path, str) and path and os.path.isfile(path):
        return path
    encoded = job_input.get("audio_prompt") or job_input.get("reference_audio")
    if isinstance(encoded, str) and encoded.strip():
        raw = base64.b64decode(encoded)
        if not raw or len(raw) > MAX_BYTES:
            raise ValueError("referencia invalida")
        with open(dest, "wb") as handle:
            handle.write(raw)
        return dest
    return None


def _download(url: str, dest: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) < 44 or len(data) > MAX_BYTES:
        raise ValueError("referencia invalida")
    with open(dest, "wb") as handle:
        handle.write(data)
