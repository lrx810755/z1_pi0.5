"""Remove only old, unreferenced Fast DDS temporary shared-memory data files."""
import argparse,json,os,re,time
from pathlib import Path

ROOT=Path('/dev/shm')
def references():
    used=set();unreadable=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
        except FileNotFoundError:continue
        try:
            content=(proc/'maps').read_text()
            used.update(re.findall(r'/dev/shm/(fastrtps_[^\s]+)',content))
        except FileNotFoundError:continue
        except PermissionError:unreadable.append(int(proc.name))
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target.startswith('/dev/shm/fastrtps_'):used.add(target.rsplit('/',1)[-1].removesuffix(' (deleted)'))
                except (FileNotFoundError,PermissionError):pass
        except (FileNotFoundError,PermissionError):pass
    return used,unreadable

def non_dds_protected(pid):
    # Standard PAM/SSH authentication daemons do not load ROS/Fast DDS.
    try:return (Path('/proc')/str(pid)/'comm').read_text().strip() in ('(sd-pam)','ssh-agent')
    except FileNotFoundError:return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    used,unreadable=references();candidates=[];now=time.time()
    # Keep recent segments and every file referenced by a running process.
    for path in ROOT.glob('fastrtps_*'):
        if not re.fullmatch(r'fastrtps_[0-9a-f]{16}',path.name):continue
        st=path.stat()
        if st.st_uid!=os.getuid() or now-st.st_mtime<86400 or path.name in used:continue
        candidates.append(dict(path=str(path),inode=st.st_ino,mtime_ns=st.st_mtime_ns,bytes=st.st_blocks*512))
    report=dict(old_unreferenced_segments=len(candidates),bytes=sum(r['bytes'] for r in candidates),referenced_segments=len(used),unreadable_same_user_processes=unreadable,applied=a.apply,removed=[])
    if a.apply:
        # Conservatively abort if same-user live processes could not be inspected.
        if any(not non_dds_protected(pid) for pid in unreadable):raise RuntimeError('Could not inspect a possible DDS process: '+str(unreadable))
        used,unreadable=references()
        if any(not non_dds_protected(pid) for pid in unreadable):raise RuntimeError('Process-reference recheck incomplete')
        for row in candidates:
            path=Path(row['path'])
            try:st=path.stat()
            except FileNotFoundError:continue
            if (st.st_ino,st.st_mtime_ns)!=(row['inode'],row['mtime_ns']) or path.name in used:continue
            path.unlink();report['removed'].append(row)
    destination=Path('/home/lp531/z1_queue_logs/reverse8_queue_20261112')/('dds_cleanup_applied.json' if a.apply else 'dds_cleanup_preview.json')
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='removed'},ensure_ascii=False,indent=2));print('removed',len(report['removed']))
if __name__=='__main__':main()
