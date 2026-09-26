import io
import unittest
import wave

import numpy as np

from encode_wav import wav_bytes


class EncodeWavTest(unittest.TestCase):
    def test_writes_16_bit_mono_and_clips(self):
        data = wav_bytes(np.array([[0.0, 0.5, -0.5, 2.0]], dtype=np.float32), 24000)
        with wave.open(io.BytesIO(data)) as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 24000)
            frames = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        self.assertEqual(frames.tolist(), [0, 16383, -16383, 32767])


if __name__ == "__main__":
    unittest.main()
