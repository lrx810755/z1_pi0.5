"""Run the independent audit after the authorized collection reaches its quota."""
import datetime
import json
from pathlib import Path
import time

from audit_saved import audit
from protocol import QUOTAS, atomic_json

RUN = Path('/home/lp531/z1_queue_logs/bottle_correction60_20261111')


def main():
    while True:
        try:
            status = json.loads((RUN / 'status.json').read_text())
        except (OSError, ValueError):
            time.sleep(5)
            continue
        if status.get('stage') == 'complete' and status.get('completed') == sum(QUOTAS.values()):
            result = audit()
            if not result['complete'] or not result['verified']:
                raise RuntimeError('Saved episodes did not pass complete independent audit')
            atomic_json(RUN / 'completion_verified.json', {
                'completed_at': datetime.datetime.now().astimezone().isoformat(),
                'verified': True, 'accepted': result['accepted'],
                'by_kind': result['by_kind'], 'quotas': QUOTAS,
                'audit': str(RUN / 'saved_integrity.json'),
                'chunk_size': 30, 'training_started': False,
            })
            return
        if status.get('stage') in ('error', 'needs_investigation', 'interrupted', 'attempt_limit'):
            raise RuntimeError('Collection stopped before completion: ' + str(status))
        time.sleep(5)


if __name__ == '__main__':
    main()
