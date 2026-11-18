from runner import command, status
from dataset_prepare import write
from config import RUN
import fcntl,json
with (RUN/'.run.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        command('verifying_resume_data','verify_resume.py')
        command('training','resume_train.py','--train')
        command('verifying_final_checkpoint','verify.py','--reload-final')
        status('completed',**json.loads((RUN/'completed.json').read_text()))
    except Exception as exc:
        status('failed',error=str(exc))
        raise
