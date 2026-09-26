import os
import tempfile
import unittest

from voice_cache import VoiceCache


class FakeModel:
    def __init__(self, fail=False):
        self.conds = None
        self.prepared = []
        self.fail = fail

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        if self.fail:
            raise RuntimeError("amostra ilegível")
        self.prepared.append(wav_fpath)
        self.conds = ("voz", wav_fpath, len(self.prepared))


class VoiceCacheTest(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"RIFF")
            self.sample = handle.name
        self.default_calls = 0

    def tearDown(self):
        os.remove(self.sample)

    def default_voice(self):
        self.default_calls += 1
        return self.sample

    def clone(self, key=None):
        job = {"audio_prompt_path": self.sample}
        if key:
            job["reference_cache_key"] = key
        return job

    def test_same_key_skips_download_and_preparation(self):
        cache = VoiceCache(self.default_voice)
        model = FakeModel()
        self.assertTrue(cache.use(model, self.clone("slot-1.wav@1"), 0.5))
        first = model.conds
        model.conds = None
        # Um link que não abre prova que a frase seguinte não baixa a amostra.
        job = {"reference_audio_url": "https://invalido.invalid/voz.wav", "reference_cache_key": "slot-1.wav@1"}
        self.assertTrue(cache.use(model, job, 0.6))
        self.assertIs(model.conds, first)
        self.assertEqual(len(model.prepared), 1)

    def test_new_sample_prepares_again(self):
        cache = VoiceCache(self.default_voice)
        model = FakeModel()
        cache.use(model, self.clone("slot-1.wav@1"), 0.5)
        cache.use(model, self.clone("slot-1.wav@2"), 0.5)
        self.assertEqual(len(model.prepared), 2)

    def test_default_voice_is_prepared_once_and_not_cloned(self):
        cache = VoiceCache(self.default_voice)
        model = FakeModel()
        self.assertFalse(cache.use(model, {"text": "Olá"}, 0.5))
        self.assertFalse(cache.use(model, {"text": "Olá de novo"}, 0.5))
        self.assertEqual(len(model.prepared), 1)
        self.assertEqual(self.default_calls, 1)

    def test_reference_without_key_is_not_kept(self):
        cache = VoiceCache(self.default_voice)
        model = FakeModel()
        self.assertTrue(cache.use(model, self.clone(), 0.5))
        self.assertTrue(cache.use(model, self.clone(), 0.5))
        self.assertEqual(len(model.prepared), 2)

    def test_oldest_voice_leaves_when_full(self):
        cache = VoiceCache(self.default_voice, limit=2)
        model = FakeModel()
        for key in ("a", "b", "c"):
            cache.use(model, self.clone(key), 0.5)
        cache.use(model, self.clone("a"), 0.5)
        self.assertEqual(len(model.prepared), 4)
        cache.use(model, self.clone("c"), 0.5)
        self.assertEqual(len(model.prepared), 4)

    def test_failed_preparation_keeps_nothing(self):
        cache = VoiceCache(self.default_voice)
        with self.assertRaises(RuntimeError):
            cache.use(FakeModel(fail=True), self.clone("k"), 0.5)
        model = FakeModel()
        self.assertTrue(cache.use(model, self.clone("k"), 0.5))
        self.assertEqual(len(model.prepared), 1)


if __name__ == "__main__":
    unittest.main()
