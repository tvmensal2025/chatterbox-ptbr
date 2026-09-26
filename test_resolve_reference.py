import base64
import os
import tempfile
import unittest

from resolve_reference import materialize_reference


class ResolveReferenceTest(unittest.TestCase):
    def test_keeps_a_local_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"RIFF")
            path = handle.name
        try:
            self.assertEqual(materialize_reference({"audio_prompt_path": path}, path + ".out"), path)
        finally:
            os.remove(path)

    def test_writes_base64_audio(self):
        dest = tempfile.mktemp(suffix=".wav")
        try:
            found = materialize_reference({"audio_prompt": base64.b64encode(b"RIFFxxxx").decode()}, dest)
            self.assertEqual(found, dest)
            with open(dest, "rb") as handle:
                self.assertTrue(handle.read().startswith(b"RIFF"))
        finally:
            if os.path.exists(dest):
                os.remove(dest)

    def test_without_audio_returns_none(self):
        self.assertIsNone(materialize_reference({"text": "Olá"}, "unused.wav"))


if __name__ == "__main__":
    unittest.main()
