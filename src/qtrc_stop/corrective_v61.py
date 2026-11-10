"""v6.1 corrective audits on frozen LOSO models; no classifier retraining.

CP9 recomputes pooled 5-class metrics per seed from every outer test event.
CP10 audits each epsilon and paired gate deletion; never tunes against outer tests.
CP11 audits matched detector onset and explicit limits (no inferred classification).
Atomic output and resume per fold/seed. Existing CP0-CP8 artifacts are read only.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from .checkpoint import atomic_dataframe_csv, atomic_write_json
from .config import load_config
from .taxonomy import SPECIES
from .stopping import Policy, apply_policy, prepare_policy_data, _decisions_from_data
from .reviewer_extension import _load_fold, _policy_for_fold, _decisions_without_gate, _metrics

GATES=['full','confidence','posterior_stability','prediction_persistence','sbr','f0_change','harmonic_ratio','harmonic_energy','stability_availability']

def _safe_csv(p):
    if not p.exists() or p.stat().st_size<20: return None
    try:return pd.read_csv(p)
    except Exception:return None

def _fold_dirs(root):
    folds=sorted((root/'cp4_predictions').glob('outer_*'))
    if len(folds)!=33:raise RuntimeError('Full 33-fold outer LOSO CP4 directory is required')
    return folds

def _loss_for_decisions(df,loss):
    lut={v:i for i,v in enumerate(SPECIES)}
    true=df.species.astype(str).to_numpy()
    return np.asarray([loss[lut[a],lut[b]]-loss[lut[a],lut[c]] for a,b,c in zip(true,df.pred_species.astype(str),df.full_pred_species.astype(str))])

def _policy_from_row(row,template,eps):
    d={k:getattr(template,k) for k in Policy.__dataclass_fields__}
    aliases={'confidence':'confidence','js_max':'js_max','persistence':'persistence', 'sbr_min_db':'sbr_min_db', 'f0_delta_max_hz':'f0_delta_max_hz'}
    for col,attr in aliases.items():
        if pd.notna(row[col]):d[attr]=int(row[col]) if col=='persistence' else float(row[col])
    d['epsilon']=float(eps)
    for a,b in [('certified','certification_passed'),('fallback','fallback_full_duration')]:
        d[b]=str(row[a]).lower() in ('true','1')
    if 'certification_ucb' in row and pd.notna(row['certification_ucb']):d['calibration_ucb_gap']=float(row['certification_ucb'])
    return Policy(**d)

def cp9(root,out,epsilon,rows_path):
    seeds=_safe_csv(rows_path)
    if seeds is None:raise FileNotFoundError(f'Missing original CP9 fold/seed results: {rows_path}')
    required={'fold','seed','confidence','js_max','persistence','sbr_min_db','f0_delta_max_hz','certified','fallback'}
    if required-set(seeds):raise ValueError(f'CP9 missing columns: {required-set(seeds)}')
    if seeds.duplicated(['fold','seed']).any():raise ValueError('Duplicate seed/fold pairs')
    all_seeds=sorted(seeds.seed.unique());out.mkdir(parents=True,exist_ok=True)
    expected={(i,s) for i in range(33) for s in all_seeds}
    if set(zip(seeds.fold,seeds.seed))!=expected:raise ValueError('Incomplete 33-fold x seed grid; no pooled report')
    for fd in _fold_dirs(root):
        fold=int(fd.name.split('_')[-1]);_,test,loss=_load_fold(root,fd)
        template=_policy_for_fold(root,fold,epsilon)
        for rec in seeds.loc[seeds.fold==fold].to_dict('records'):
            sid=int(rec['seed']);target=out/'folds'/f'outer_{fold:03d}'/f'seed_{sid:03d}.csv.gz'
            if target.exists() and target.stat().st_size>100:continue
            policy=_policy_from_row(rec,template,epsilon)
            dec=apply_policy(test,sorted(test.duration_sec.unique().astype(float)),policy)
            gap=_loss_for_decisions(dec,loss)
            dec['taxonomic_gap']=gap
            dec['fold']=fold;dec['seed']=sid
            dec['certified']=bool(policy.certification_passed)
            dec['fallback']=bool(policy.fallback_full_duration)
            target.parent.mkdir(parents=True,exist_ok=True)
            # atomic gzip dataframe: write temp then rename
            tmp=target.with_suffix('.csv.gz.tmp')
            dec.to_csv(tmp,index=False,compression='gzip');tmp.replace(target)
    summaries=[];species=[]
    for seed in all_seeds:
        paths=[out/'folds'/f'outer_{f:03d}'/f'seed_{int(seed):03d}.csv.gz' for f in range(33)]
        if not all(p.exists() for p in paths):raise RuntimeError('Incomplete seed, no summary')
        a=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        if a.event_id.duplicated().any():raise RuntimeError(f'Overlapping outer-test event IDs, seed={seed}')
        y=a.species.astype(str);pred=a.pred_species.astype(str)
        ps=a.groupby('subject_id').taxonomic_gap.mean()
        p,r,f,n=precision_recall_fscore_support(y,pred,labels=SPECIES,zero_division=0)
        for i,s in enumerate(SPECIES):species.append(dict(seed=int(seed),species=s,events=int(n[i]),precision=p[i],recall=r[i],f1=f[i]))
        summaries.append(dict(seed=int(seed),events=len(a),subjects=a.subject_id.nunique(),
          accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=SPECIES,average='macro',zero_division=0),
          mean_stop_sec=a.stop_duration_sec.mean(),subject_mean_gap=ps.mean(),
          certification_rate=a.groupby('fold').certified.first().mean(),fallback_rate=a.groupby('fold').fallback.first().mean(),
          at_full_fraction=np.isclose(a.stop_duration_sec,2.0).mean()))
    table=pd.DataFrame(summaries)
    atomic_dataframe_csv(table,out/'cp9_pooled_by_seed.csv')
    atomic_dataframe_csv(pd.DataFrame(species),out/'cp9_pooled_species_by_seed.csv')
    quant=[]
    for col in table.columns.difference(['seed']):
        if pd.api.types.is_numeric_dtype(table[col]):
            v=table[col];quant.append(dict(metric=col,mean=v.mean(),sd=v.std(ddof=1),p05=v.quantile(.05),median=v.median(),p95=v.quantile(.95)))
    atomic_dataframe_csv(pd.DataFrame(quant),out/'cp9_pooled_summary.csv')
    atomic_write_json(out/'status.json',dict(complete=True,seeds=len(all_seeds),outer_folds=33,interpretation='Pooled Macro-F1 from all disjoint outer-test events, not foldwise Macro-F1. Split-seed percentiles are sensitivity summaries, not coverage guarantees.'))

def cp10(root,out,epsilons,nominal=0.05):
    out.mkdir(parents=True,exist_ok=True)
    for eps in epsilons:
        tag=f'e{eps:.3f}'
        for fd in _fold_dirs(root):
            fold=int(fd.name.split('_')[-1]);dest=out/tag/'folds'/f'outer_{fold:03d}.csv.gz'
            if dest.exists() and dest.stat().st_size>100:continue
            _,test,loss=_load_fold(root,fd)
            pol=_policy_for_fold(root,fold,eps)
            data=prepare_policy_data(test,sorted(test.duration_sec.unique().astype(float)))
            frames=[]
            full=_decisions_from_data(data,pol,False)
            for gate in GATES:
                dec=full if gate=='full' else _decisions_without_gate(data,pol,gate)
                b=dec[['event_id','subject_id','species','pred_species','full_pred_species','stop_duration_sec']].copy()
                b['taxonomic_gap']=_loss_for_decisions(dec,loss)
                b['gate']=gate;b['fold']=fold;b['epsilon']=eps
                base=full.set_index('event_id');b['different_from_full']=(b.pred_species.to_numpy()!=full.pred_species.to_numpy()) | (~np.isclose(b.stop_duration_sec.to_numpy(),full.stop_duration_sec.to_numpy()))
                frames.append(b)
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.gz.tmp');pd.concat(frames).to_csv(tmp,index=False,compression='gzip');tmp.replace(dest)
        paths=sorted((out/tag/'folds').glob('outer_*.csv.gz'))
        if len(paths)!=33:raise RuntimeError(f'Incomplete CP10 epsilon {eps}')
        a=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        rows=[];perfold=[]
        for name,g in a.groupby('gate'):
            if g.event_id.duplicated().any():raise RuntimeError(f'CP10 duplicate event in {name} eps {eps}')
            rows.append(dict(epsilon=eps,gate=name,events=len(g),subjects=g.subject_id.nunique(),
              accuracy=accuracy_score(g.species,g.pred_species),macro_f1=f1_score(g.species,g.pred_species,labels=SPECIES,average='macro',zero_division=0),
              mean_stop_sec=g.stop_duration_sec.mean(),subject_mean_gap=g.groupby('subject_id').taxonomic_gap.mean().mean(),
              changed_events=int(g.different_from_full.sum()),changed_folds=int(g.groupby('fold').different_from_full.any().sum())))
            for fold,f in g.groupby('fold'):
                perfold.append(dict(epsilon=eps,gate=name,fold=int(fold),events=len(f),changed_events=int(f.different_from_full.sum()),mean_stop_sec=f.stop_duration_sec.mean(),accuracy=accuracy_score(f.species,f.pred_species)))
        atomic_dataframe_csv(pd.DataFrame(rows),out/tag/'cp10_pooled_gate_metrics.csv')
        atomic_dataframe_csv(pd.DataFrame(perfold),out/tag/'cp10_fold_gate_metrics.csv')
    big=pd.concat([pd.read_csv(out/f'e{eps:.3f}'/'cp10_pooled_gate_metrics.csv') for eps in epsilons])
    atomic_dataframe_csv(big,out/'cp10_all_epsilons_pooled.csv')
    atomic_write_json(out/'status.json',dict(complete=True,epsilons=epsilons,interpretation='Frozen CP5 policies; one-gate deletion on identical outer-test events. Not a trained simplified policy; no superiority claim from test-guided selection.'))

def cp11_audit(root,out):
    """Checks event matching coverage without inventing classification on shifted windows."""
    src=root/'reviewer_extension_v6'/'cp11_detector_sensitivity'/'cp11_event_matching.csv'
    tab=_safe_csv(src)
    if tab is None:raise FileNotFoundError(f'Missing CP11 event-matching CSV: {src}')
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    for threshold,g in tab.groupby('threshold'):
        q=g[g.matched.astype(str).str.lower().isin(['true','1'])]
        v=q.onset_shift_sec.abs().dropna()
        report.append(dict(threshold=threshold,proposed_events=len(g),matched_events=len(q),unmatched_events=len(g)-len(q),
          matched_fraction=len(q)/max(len(g),1),mean_abs_onset_shift_sec=v.mean(),median_abs_onset_shift_sec=v.median(),p95_abs_onset_shift_sec=v.quantile(.95),
          classification_recomputed=False))
    atomic_dataframe_csv(pd.DataFrame(report),out/'cp11_matching_audit.csv')
    atomic_write_json(out/'status.json',dict(complete=True,detector_sensitivity_complete=True,end_to_end_classification_complete=False,
      limitation='Shifted onset windows require re-extraction and inference from original recordings with frozen CP4 fold-specific models and matched event IDs. Nominal CP5 predictions cannot substitute for changed windows.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--out',default=None)
    p.add_argument('--epsilon',type=float,default=0.05);p.add_argument('--epsilons',nargs='+',type=float,default=[.02,.05,.10,.15]);p.add_argument('--only',choices=['cp9','cp10','cp11','all'],default='all')
    a=p.parse_args();root=Path(a.run_dir).resolve();out=Path(a.out).resolve() if a.out else root/'reviewer_extension_v61'
    out.mkdir(parents=True,exist_ok=True)
    if a.only in ('cp9','all'):cp9(root,out/'cp9_corrected',a.epsilon,root/'reviewer_extension_v6'/'cp9_split_stability'/'cp9_split_seed_fold_metrics.csv')
    if a.only in ('cp10','all'):cp10(root,out/'cp10_multi_epsilon',a.epsilons)
    if a.only in ('cp11','all'):cp11_audit(root,out/'cp11_audit')
    status={t:(out/t/'status.json').exists() for t in ['cp9_corrected','cp10_multi_epsilon','cp11_audit']}
    atomic_write_json(out/'completion.json',status)
    print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()
