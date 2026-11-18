"""Wait for evaluator completion using a kernel exit event; no trial polling."""
import datetime,json,os,select,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent

def save(stage,**values):
    p=R/'report_completion_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**values),ensure_ascii=False,indent=2))
    tmp.replace(p)

def main():
    pid=int(sys.argv[1]);save('waiting_for_runner_exit',runner_pid=pid,event_wait='pidfd; no progress polling')
    try:
        fd=os.pidfd_open(pid)
    except ProcessLookupError:
        fd=None
    if fd is not None:
        try:select.select([fd],[],[])
        finally:os.close(fd)
    completed=R/'completed.json'
    if not completed.exists():
        raise RuntimeError('Runner exited without a completed30-trial summary; original records retained')
    summary=json.loads(completed.read_text())
    if not(summary.get('completed') and summary.get('total')==30):
        raise RuntimeError('Runner summary does not confirm all30 trials')
    save('generating_report')
    with (R/'report_generation.log').open('a') as log:
        subprocess.run(['/usr/bin/python3',str(R/'summarize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    save('completed',total=30,successes=summary['successes'],report=str(R/'验证报告.md'))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('failed',error=str(exc));raise
