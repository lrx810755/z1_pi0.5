"""A failed official transfer must retain bytes and fall back to the mirror."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import download_base


class DownloadTest(unittest.TestCase):
    def test_fallback_resume_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            calls = []

            def transfer(command, **kwargs):
                url = command[-3]
                partial = Path(command[-1])
                calls.append(url)
                self.assertIn('--continue-at', command)
                if len(calls) <= 3:
                    partial.write_bytes(b'abc')
                    raise subprocess.CalledProcessError(28, command)
                self.assertEqual(partial.read_bytes(), b'abc')
                partial.write_bytes(b'abcdef')

            with patch.object(download_base, 'BASE', base), patch.object(subprocess, 'run', transfer), patch.object(download_base.time, 'sleep'):
                download_base.download('model.safetensors', hashlib.sha256(b'abcdef').hexdigest())
            self.assertIn('huggingface.co/', calls[0])
            self.assertIn('hf-mirror.com/', calls[3])
            self.assertEqual((base / 'model.safetensors').read_bytes(), b'abcdef')
            self.assertFalse((base / 'model.safetensors.http-download').exists())


if __name__ == '__main__':
    unittest.main()
