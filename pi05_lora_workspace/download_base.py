"""Resume official weights, falling back to a domestic mirror; verify exact SHA256."""
import hashlib
from pathlib import Path
import subprocess
import time

from safetensors import safe_open

WORK = Path(__file__).resolve().parent
BASE = WORK / 'weights/pi05_base'
ORIGINS = ['https://huggingface.co', 'https://hf-mirror.com']
REVISION = 'b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba'
SHA256 = '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(filename, expected_sha=None):
    target = BASE / filename
    if target.exists():
        if expected_sha and checksum(target) != expected_sha:
            raise ValueError(f'Existing {filename} checksum mismatch')
        return
    partial = BASE / f'{filename}.http-download'
    for origin in ORIGINS:
        url = f'{origin}/lerobot/pi05_base/resolve/{REVISION}/{filename}'
        for attempt in range(1, 4):
            offset = partial.stat().st_size if partial.exists() else 0
            print(f'DOWNLOAD_SOURCE {origin} {filename}; attempt={attempt}; resume_bytes={offset}', flush=True)
            try:
                # A new curl invocation recalculates the resume offset. curl's own retry
                # reuses its initial offset and discards progress made during that request.
                subprocess.run(['curl', '--http1.1', '--fail', '--location', '--continue-at', '-',
                                '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '90',
                                '--silent', '--show-error', url, '-o', str(partial)], check=True)
                if expected_sha and checksum(partial) != expected_sha:
                    partial.unlink()
                    raise ValueError(f'{filename} checksum mismatch')
                partial.rename(target)
                return
            except (subprocess.CalledProcessError, ValueError) as exc:
                print(f'SOURCE_FAILED {origin}: {type(exc).__name__}; preserving progress', flush=True)
                if attempt < 3:
                    time.sleep(5)
    raise RuntimeError(f'All download sources failed for {filename}; partial download retained')


if __name__ == '__main__':
    BASE.mkdir(parents=True, exist_ok=True)
    for filename in ['config.json', 'policy_preprocessor.json', 'policy_postprocessor.json', 'README.md']:
        download(filename)
    download('model.safetensors', SHA256)
    with safe_open(BASE / 'model.safetensors', framework='pt', device='cpu') as weights:
        keys = weights.keys()
        assert len(keys) > 500
        assert any('action_out_proj' in key for key in keys)
    print('BASE_WEIGHTS_VERIFIED SHA256=' + SHA256, flush=True)
