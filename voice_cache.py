"""Guarda a voz já preparada. A mesma amostra não é baixada nem preparada de novo a cada frase."""
import os
import tempfile
from collections import OrderedDict
from typing import Callable

from resolve_reference import materialize_reference

DEFAULT_KEY = ":voz-padrao"
MAX_KEY = 300


class VoiceCache:
    def __init__(self, default_voice: Callable[[], str], limit: int = 8):
        self._default_voice = default_voice
        self._limit = limit
        self._voices: OrderedDict[str, object] = OrderedDict()

    def use(self, model, job_input: dict, exaggeration: float) -> bool:
        """Deixa a voz do pedido em model.conds e diz se ela é clonada."""
        key = str(job_input.get("reference_cache_key") or "")[:MAX_KEY]
        if key and self._reuse(model, key):
            return True
        fd, dest = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audio_path = materialize_reference(job_input, dest)
            if audio_path is None:
                if self._reuse(model, DEFAULT_KEY):
                    return False
                key, audio_path = DEFAULT_KEY, self._default_voice()
            model.prepare_conditionals(audio_path, exaggeration=exaggeration)
        finally:
            if os.path.exists(dest):
                os.remove(dest)
        if key:
            self._voices[key] = model.conds
            while len(self._voices) > self._limit:
                self._voices.popitem(last=False)
        return key != DEFAULT_KEY

    def _reuse(self, model, key: str) -> bool:
        conds = self._voices.get(key)
        if conds is None:
            return False
        self._voices.move_to_end(key)
        model.conds = conds
        return True
