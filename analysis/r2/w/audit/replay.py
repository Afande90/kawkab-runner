from pathlib import Path
import zipfile,shutil,hashlib,sys,subprocess,time,datetime,json
here=Path(__file__).resolve().parent
workspace=here.parent
dest=here/'rerun'
e=here/'evidence'
sha=lambda b:hashlib.sha256(b).hexdigest()
z=zipfile.ZipFile(workspace/'JSpace-Predict-Then-Train.zip')
for name in z.namelist():
 assert z.read(name)==(workspace/name).read_bytes(),name
 if name.endswith(('.py','.tsv','.txt')):
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
for name in ['intact','jinstall','capacity']:
 p=dest/name
 if not p.exists():p.symlink_to(workspace/name,target_is_directory=True)
print('Archived files verified:',len(z.namelist()),flush=True)
with (e/'replay.txt').open('w') as transcript:
 for s,log,output in [('association/trial.py','run.log','result.json'),('association/repeat/trial.py','run.log','result.json'),('association/transfer.py','transfer.log','transfer.json'),('association/repeat/transfer.py','transfer.log','transfer.json'),('functional/test.py','run.log','answers.csv')]:
    path=Path(s);t=time.perf_counter()
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    proc=subprocess.run([sys.executable,'-u',str(dest/s)],capture_output=True)
    elapsed=time.perf_counter()-t
    (dest/path.parent/log).write_bytes(proc.stdout)
    (e/(s.replace('/','_')+'.stderr')).write_bytes(proc.stderr)
    print(f'{s}  start_UTC={start}  elapsed_s={elapsed:.3f}  exit={proc.returncode}',file=transcript)
    print(proc.stdout.decode(),file=transcript,end='')
    if proc.returncode:raise RuntimeError(proc.stderr.decode())
    for n in [log,output]:
        key=str(path.parent/n)
        before=z.read(key);after=(dest/key).read_bytes()
        print(f'{key}  original={sha(before)}  rerun={sha(after)}  identical={before==after}',file=transcript)
    transcript.flush()
    print(s,round(elapsed,3),flush=True)
 for name in ('association','association/repeat'):
    d=json.loads((dest/name/'transfer.json').read_text())
    for k in ['before','after','no_copy']:
        v=[x for arm in d.values() for x in arm['small'][k]]
        print(f'{name} {k} correct={sum(x>0 for x in v)}/{len(v)}',file=transcript)
