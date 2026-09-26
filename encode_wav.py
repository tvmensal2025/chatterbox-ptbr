"""WAV de 16 bits: metade do tamanho do float32, que é o que o torchaudio grava por padrão."""
import io
import wave

import numpy as np


def wav_bytes(samples, sample_rate: int) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()
