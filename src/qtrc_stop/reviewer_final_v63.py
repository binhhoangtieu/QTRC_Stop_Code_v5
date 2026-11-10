"""Reviewer final validation: preflight, controlled CP12-14, strict verification, evidence ZIP.
No training and no changes to CP0-CP11. CP12-14 are run in v62 output layout.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from . import revision_v62 as rv

NEEDED = {
'CP12':('cp12_metric_audit/cp12_pooled_recomputed.csv','cp12_metric_audit/cp12_metric_reconciliation.csv','cp12_metric_audit/status.json'),
'CP13':('cp13_training_only_simplification/cp13_pooled_metrics.csv','cp13_training_only_simplification/cp13_fold_choices.csv','cp13_training_only_simplification/status.json'),
'CP14':('cp14_matched_event_inference/cp14_matched_metrics.csv','cp14_matched_event_inference/cp14_paired_comparison.csv.gz','cp14_matched_event_inference/status.json'),
}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def preflight(root,out):
    rows=[]
    for stage,pattern,mincount in [('CP4','cp4_predictions/outer_*',33),('CP6','cp6_results/summary_metrics.csv',1),('Config','resolved_config.yaml',1),('CP11','reviewer_extension_v6/cp11_detector_sensitivity/cp11_event_matching.csv',1)]:
        matches=list(root.glob(pattern)); rows.append(dict(stage=stage,requirement=pattern,found=len(matches),minimum=mincount,pass_check=len(matches)>=mincount))
    folds=sorted((root/'cp4_predictions').glob('outer_*'))
    modelcount=sum(len(list(x.glob('models/*.joblib'))) for x in folds)
    row=dict(stage='CP14',requirement='prefix_models',found=modelcount,minimum=7*33,pass_check=modelcount>=7*33); rows.append(row)
    missing_paths=0
    ev=root/'cp1_events/events.csv'
    if ev.exists():
        df=pd.read_csv(ev,usecols=['path']); missing_paths=sum(not Path(str(p)).exists() for p in df.path.drop_duplicates());rows.append(dict(stage='CP14',requirement='source_recording_paths',found=df.path.nunique()-missing_paths,minimum=df.path.nunique(),pass_check=missing_paths==0))
    else: rows.append(dict(stage='CP14',requirement='cp1_events/events.csv',found=0,minimum=1,pass_check=False))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'preflight.csv',index=False)
    return rows

def verify(root,out):
    rows=[]
    for stage, names in NEEDED.items():
        for rel in names:
            p=out/rel; rows.append(dict(checkpoint=stage,file=rel,exists=p.is_file(),size_bytes=p.stat().st_size if p.exists() else 0))
    statuses={}
    for st, names in NEEDED.items():
        p=out/names[-1];statuses[st]=json.loads(p.read_text()) if p.exists() else {}
    reconciled=statuses['CP12'].get('reconciled') is True
    complete=all(r['exists'] and r['size_bytes']>0 for r in rows) and all(s.get('complete') is True for s in statuses.values())
    fold13=out/'cp13_training_only_simplification/cp13_fold_choices.csv'
    if fold13.exists():
        choices=pd.read_csv(fold13)
        complete &= len(choices)==33 and choices.fold.nunique()==33
    report={'complete':bool(complete),'metric_reconciled':reconciled,'reviewer_evidence_ready':bool(complete and reconciled),
            'cp12':statuses['CP12'],'cp13':statuses['CP13'],'cp14':statuses['CP14'],
            'notice':'Even when ready, claims must be reviewed against the data and manuscript.'}
    pd.DataFrame(rows).to_csv(out/'v63_file_checklist.csv',index=False)
    tmp=out/'v63_verification.json.part';tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False));os.replace(tmp,out/'v63_verification.json')
    return report

def export(root,out,dest):
    dest.parent.mkdir(parents=True,exist_ok=True); paths=[]
    for base in [root/'cp6_results',root/'cp8_reports',root/'reviewer_extension_v6',root/'reviewer_extension_v61',out]:
        if not base.is_dir(): continue
        for p in base.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in ['.csv','.json','.yaml','.txt','.tex','.gz']:continue
            if p.stat().st_size>75*1024*1024:continue
            paths.append(p)
    for f in ('resolved_config.yaml','run_meta.json','state.json'):
        p=root/f
        if p.exists():paths.append(p)
    paths=list(dict.fromkeys(paths));manifest=[];temp=dest.with_name(dest.name+'.part')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in paths:
            relative=p.relative_to(root)
            z.write(p,relative.as_posix());manifest.append(dict(file=str(relative),bytes=p.stat().st_size,sha256=sha(p)))
        z.writestr('EVIDENCE_MANIFEST.json',json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'files':manifest},indent=2))
    os.replace(temp,dest)
    return len(paths)

def main():
    ap=argparse.ArgumentParser(description='QTRC-Stop CP12/CP13/CP14 reviewer evidence final validation')
    ap.add_argument('task',choices=['preflight','audit','simplify','onset','verify','export','all'])
    ap.add_argument('--run-dir',required=True);ap.add_argument('--epsilon',type=float,default=.05)
    ap.add_argument('--seed',type=int,default=20261016);ap.add_argument('--bootstrap-repeats',type=int,default=2000)
    ap.add_argument('--max-pairs',type=int,default=None,help='Smoke test only, must not use for manuscript')
    a=ap.parse_args();root=Path(a.run_dir).expanduser().resolve();out=root/'reviewer_extension_v63'
    if not root.exists():ap.error(f'Run directory not found: {root}')
    try:
        if a.task in ('preflight','all'):
            r=preflight(root,out)
            for x in r:print(x)
            if a.task=='all' and not all(x['pass_check'] for x in r):
                raise RuntimeError('Preflight missing essential artifacts; see preflight.csv')
        if a.task in ('audit','all'):rv.audit(root,out,a.epsilon)
        if a.task in ('simplify','all'):rv.simplify(root,out,a.epsilon,a.seed,a.bootstrap_repeats)
        if a.task in ('onset','all'):rv.onset(root,out,a.epsilon,max_pairs=a.max_pairs)
        if a.task in ('verify','all'):
            r=verify(root,out);print(json.dumps(r,indent=2,ensure_ascii=False))
        if a.task in ('export','all'):
            dest=root.parent/f'{root.name}_reviewer_v63_evidence.zip';n=export(root,out,dest)
            print(f'Exported {n} evidence files to {dest}')
    except Exception as e:
        print(f'FAILED: {type(e).__name__}: {e}',file=sys.stderr)
        raise
if __name__=='__main__': main()
