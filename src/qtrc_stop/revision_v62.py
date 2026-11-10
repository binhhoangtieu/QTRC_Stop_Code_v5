"""TIM reviewer v6.2. Read-only CP0--CP8; outputs are atomic and resumable.

Commands: audit, simplify, onset, verify. Every reported metric comes from
actual outer-test decisions; selection uses calibrated OOF training subjects.
"""
from __future__ import annotations
import argparse, json, hashlib, os, tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .taxonomy import SPECIES
from .reviewer_extension import _load_fold, _policy_for_fold, _decisions_without_gate
from .stopping import apply_policy, prepare_policy_data, _decisions_from_data
from .checkpoint import atomic_dataframe_csv, atomic_write_json

DROPS=['full','posterior_stability','prediction_persistence','sbr','f0_change','harmonic_ratio','harmonic_energy','stability_availability']

def folds(root):
    f=sorted((root/'cp4_predictions').glob('outer_*'))
    if len(f)!=33:raise RuntimeError(f'Expected 33 CP4 folds, got {len(f)}')
    return f

def durations(df):return sorted(map(float,df.duration_sec.unique()))

def _loss(df,L):
    idx={s:i for i,s in enumerate(SPECIES)}
    return np.asarray([L[idx[a],idx[b]]-L[idx[a],idx[c]] for a,b,c in zip(df.species,df.pred_species,df.full_pred_species)],float)

def _dec(pred,policy,drop='full'):
    d=prepare_policy_data(pred,durations(pred))
    if drop=='full':return _decisions_from_data(d,policy,False)
    return _decisions_without_gate(d,policy,drop)

def _stats(df,L,method,eps,fold=None):
    gap=_loss(df,L)
    g=pd.DataFrame({'subject_id':df.subject_id,'gap':gap}).groupby('subject_id').gap.mean()
    return dict(method=method,epsilon=eps,fold=fold,events=len(df),subjects=df.subject_id.nunique(),
       accuracy=accuracy_score(df.species,df.pred_species),
       macro_f1=f1_score(df.species,df.pred_species,labels=SPECIES,average='macro',zero_division=0),
       mean_stop_sec=df.stop_duration_sec.mean(),subject_mean_gap=g.mean(),event_mean_gap=gap.mean(),
       mean_stop_at_2s=np.isclose(df.stop_duration_sec,2.).mean())

def _safe_write(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_dataframe_csv(df,path,compression='gzip' if str(path).endswith('.gz') else None)

def _cached_csv(path):
    if path.exists() and path.stat().st_size>100:
        try:return pd.read_csv(path)
        except Exception:pass
    return None

def audit(root,out,eps):
    folder=out/'cp12_metric_audit';folder.mkdir(parents=True,exist_ok=True)
    rows=[];fparts=[]
    for fd in folds(root):
        f=int(fd.name[-3:]);path=folder/'folds'/f'outer_{f:03d}.csv.gz'
        d=_cached_csv(path)
        if d is None:
            oof,test,L=_load_fold(root,fd)
            if set(oof.subject_id)&set(test.subject_id):raise RuntimeError(f'Training leakage outer={f}')
            pol=_policy_for_fold(root,f,eps)
            dec=apply_policy(test,durations(test),pol)
            d=dec[['event_id','subject_id','species','pred_species','full_pred_species','stop_duration_sec']].copy()
            d['fold']=f;d['gap']=_loss(dec,L)
            _safe_write(d,path)
        if d.event_id.duplicated().any():raise RuntimeError(f'Duplicate decisions outer={f}')
        rows.append(dict(fold=f,events=len(d),subjects=d.subject_id.nunique(),mean_stop_sec=d.stop_duration_sec.mean(),
             subject_mean_gap=d.groupby('subject_id').gap.mean().mean(),event_mean_gap=d.gap.mean(),
             accuracy=accuracy_score(d.species,d.pred_species)))
        fparts.append(d)
    a=pd.concat(fparts,ignore_index=True)
    if a.event_id.duplicated().any():raise RuntimeError('Duplicate pooled event IDs')
    if a.subject_id.nunique()!=33:raise RuntimeError('Expected 33 unique LOSO test subjects')
    pooled={'events':len(a),'subjects':a.subject_id.nunique(),'accuracy':accuracy_score(a.species,a.pred_species),
        'macro_f1':f1_score(a.species,a.pred_species,labels=SPECIES,average='macro',zero_division=0),
        'mean_stop_sec':a.stop_duration_sec.mean(),'subject_mean_gap':a.groupby('subject_id').gap.mean().mean(),
        'event_mean_gap':a.gap.mean(), 'subject_mean_positive_part_gap':a.assign(positive_gap=a.gap.clip(lower=0)).groupby('subject_id').positive_gap.mean().mean()}
    _safe_write(pd.DataFrame(rows),folder/'cp12_fold_recomputed.csv')
    _safe_write(pd.DataFrame([pooled]),folder/'cp12_pooled_recomputed.csv')
    checks=[]
    original=root/'cp6_results'/'summary_metrics.csv'
    if original.exists():
        tab=pd.read_csv(original)
        for _,r in tab.iterrows():
            if str(r.get('method',''))!='qtrc_stop' or abs(float(r.get('epsilon',-1))-eps)>1e-8:continue
            for col in ['accuracy','macro_f1','mean_stop_sec','subject_mean_gap']:
                if col in r and pd.notna(r[col]):
                    checks.append(dict(source='cp6_summary',metric=col,reported=float(r[col]),recomputed=float(pooled[col]),
                      absolute_difference=abs(float(r[col])-pooled[col]),within_1e_6=abs(float(r[col])-pooled[col])<1e-6))
    prior=root/'reviewer_extension_v61'/'cp10_multi_epsilon'/'cp10_all_epsilons_pooled.csv'
    if prior.exists():
        tab=pd.read_csv(prior)
        tab=tab[(np.isclose(tab.epsilon,eps)) & (tab.gate=='full')]
        for _,r in tab.iterrows():
            for col in ['accuracy','macro_f1','mean_stop_sec','subject_mean_gap']:
                if col in r:
                    checks.append(dict(source='cp10_v61_full',metric=col,reported=float(r[col]),recomputed=float(pooled[col]),
                      absolute_difference=abs(float(r[col])-pooled[col]),within_1e_6=abs(float(r[col])-pooled[col])<1e-6))
    _safe_write(pd.DataFrame(checks,columns=['source','metric','reported','recomputed','absolute_difference','within_1e_6']),folder/'cp12_metric_reconciliation.csv')
    discrepancies=[c for c in checks if not c['within_1e_6']]
    atomic_write_json(folder/'status.json',dict(complete=True,outer_folds=33,events=len(a),
         reconciled=len(discrepancies)==0,discrepancy_count=len(discrepancies),
         note='Mismatch is not corrected automatically. Confirm metric definitions, fold loss matrix, aggregation and CP6 run identity.'))

def _gate_apply(data,pol,drop):
    return _decisions_from_data(data,pol,False) if drop=='full' else _decisions_without_gate(data,pol,drop)

def _risk_subject(dec,L):
    g=_loss(dec,L)
    return pd.DataFrame({'subject':dec.subject_id,'gap':g}).groupby('subject').gap.mean().to_numpy()

def _ucb(values,rng,repeats=1000):
    if len(values)<2:return float('inf')
    x=np.asarray(values);res=np.empty(repeats)
    for k in range(repeats):res[k]=rng.choice(x,len(x),replace=True).mean()
    return float(np.quantile(res,.90))

def simplify(root,out,eps,seed=20261016,repeats=1000):
    folder=out/'cp13_training_only_simplification';folder.mkdir(parents=True,exist_ok=True)
    selected=[];summaries=[];parts=[]
    for fd in folds(root):
        f=int(fd.name[-3:]);target=folder/'folds'/f'outer_{f:03d}.csv.gz'
        tab=_cached_csv(target)
        log=folder/'folds'/f'outer_{f:03d}_selection.csv'
        choice=folder/'folds'/f'outer_{f:03d}_choice.json'
        if tab is None or not log.exists() or not choice.exists():
            oof,test,L=_load_fold(root,fd)
            if set(oof.subject_id)&set(test.subject_id):raise RuntimeError('Train/test subject leakage')
            policy=_policy_for_fold(root,f,eps)
            subjects=sorted(oof.subject_id.unique().astype(str));rng=np.random.default_rng(seed+f)
            perm=rng.permutation(subjects);n=len(perm)//2;sel=set(perm[:n]);check=set(perm[n:])
            if not sel or not check:raise RuntimeError('Empty selection/checking subjects')
            oof_sel=oof[oof.subject_id.astype(str).isin(sel)]
            oof_check=oof[oof.subject_id.astype(str).isin(check)]
            ds=prepare_policy_data(oof_sel,durations(oof_sel));dc=prepare_policy_data(oof_check,durations(oof_check))
            records=[]
            for gate in DROPS:
                a=_gate_apply(ds,policy,gate);b=_gate_apply(dc,policy,gate)
                selection_gap=float(_risk_subject(a,L).mean());selection_time=float(a.stop_duration_sec.mean())
                bound=_ucb(_risk_subject(b,L),np.random.default_rng(seed+f*100+len(records)),repeats)
                records.append(dict(gate=gate,selection_gap=selection_gap,selection_time=selection_time,
                              checking_ucb=bound,checking_pass=bound<=eps,
                              checking_subjects=len(check),selection_subjects=len(sel)))
            audit=pd.DataFrame(records)
            # Choice is determined ONLY by selection subjects among checking-approved candidates.
            eligible=audit[(audit.checking_pass)&(audit.selection_gap<=eps)]
            if eligible.empty:chosen='fallback_fixed_2s'
            else:chosen=str(eligible.sort_values(['selection_time','gate']).iloc[0].gate)
            # If CP5 already fell back, never override with a simplified rule.
            if policy.fallback_full_duration:chosen='fallback_fixed_2s'
            dt=prepare_policy_data(test,durations(test))
            full=apply_policy(test,durations(test),policy)
            if chosen=='fallback_fixed_2s':
                new=full.copy();last=durations(test)[-1]
                p=test[np.isclose(test.duration_sec,last)].set_index('event_id')
                new['stop_duration_sec']=last
                new['pred_species']=[SPECIES[int(np.argmax(p.loc[e,[f'p_{s}' for s in SPECIES]].to_numpy(dtype=float)))] for e in new.event_id]
            else:new=_gate_apply(dt,policy,chosen)
            cols=['event_id','subject_id','species','pred_species','full_pred_species','stop_duration_sec']
            tab=new[cols].copy();tab['fold']=f;tab['chosen_gate']=chosen;tab['taxonomic_gap']=_loss(new,L)
            _safe_write(tab,target);_safe_write(audit,log)
            atomic_write_json(choice,dict(fold=f,chosen_gate=chosen,outer_test_subjects=sorted(map(str,test.subject_id.unique())),
                  selection_subjects=sorted(sel),checking_subjects=sorted(check),
                  bound='90th percentile of subject-bootstrap means; empirical, not distribution-free',
                  only_training_used_for_choice=True))
        if tab.event_id.duplicated().any():raise RuntimeError(f'Duplicate simplified events outer={f}')
        ch=json.loads(choice.read_text());selected.append(dict(fold=f,chosen_gate=ch['chosen_gate'],
              selection_subjects=len(ch['selection_subjects']),checking_subjects=len(ch['checking_subjects'])))
        summaries.append(dict(fold=f,events=len(tab),accuracy=accuracy_score(tab.species,tab.pred_species),
            mean_stop_sec=tab.stop_duration_sec.mean(),subject_mean_gap=tab.groupby('subject_id').taxonomic_gap.mean().mean()))
        parts.append(tab)
    a=pd.concat(parts,ignore_index=True)
    if a.event_id.duplicated().any() or a.subject_id.nunique()!=33:raise RuntimeError('LOSO pooled integrity failed')
    pooled=dict(epsilon=eps,events=len(a),subjects=a.subject_id.nunique(),
         accuracy=accuracy_score(a.species,a.pred_species),macro_f1=f1_score(a.species,a.pred_species,labels=SPECIES,average='macro',zero_division=0),
         mean_stop_sec=a.stop_duration_sec.mean(),subject_mean_gap=a.groupby('subject_id').taxonomic_gap.mean().mean(),
         event_mean_gap=a.taxonomic_gap.mean(),at_full_fraction=np.isclose(a.stop_duration_sec,2.0).mean())
    _safe_write(pd.DataFrame(selected),folder/'cp13_fold_choices.csv')
    _safe_write(pd.DataFrame(summaries),folder/'cp13_fold_metrics.csv')
    _safe_write(pd.DataFrame([pooled]),folder/'cp13_pooled_metrics.csv')
    _safe_write(pd.DataFrame(selected).groupby('chosen_gate').size().reset_index(name='folds'),folder/'cp13_choice_distribution.csv')
    atomic_write_json(folder/'status.json',dict(complete=True,outer_folds=33,events=len(a),
       training_only=True,bootstrap_empirical=True,predefined_candidates=DROPS,
       warning='Policy gate deletion is selected on training OOF; compare against independent frozen CP6. Not a formal guarantee.'))

def onset(root,out,eps,thresholds=(.30,.40),max_pairs=None):
    """Inference on matched, shifted windows only; no fabricated detection labels."""
    from .config import load_config
    from .io import read_signal_csv
    from .signal import complex_from_amplitude_phase,preprocess_complex
    from .features import extract_event_feature_rows
    from .models import HierarchicalClassifier
    from .calibration import apply_temperature
    from .taxonomy import SPECIES

    # CP14 repeatedly evaluates the same seven frozen prefix models for many
    # matched events in a subject. Keep only one fold's seven models resident
    # to avoid re-reading large joblib files for every event without growing
    # memory with the number of folds.
    @lru_cache(maxsize=7)
    def load_model_cached(model_path: str):
        return HierarchicalClassifier.load(Path(model_path))

    src=root/'reviewer_extension_v6'/'cp11_detector_sensitivity'/'cp11_event_matching.csv'
    if not src.exists():raise FileNotFoundError(f'CP11 pairs missing: {src}')
    pairs=pd.read_csv(src)
    events=pd.read_csv(root/'cp1_events'/'events.csv')
    cfg=load_config(root/'resolved_config.yaml')
    fs=float(cfg['dataset']['sampling_rate_hz']);ds=list(map(float,cfg['features']['prefix_durations_sec']))
    fold_lookup={}
    for fd in folds(root):
        test=pd.read_csv(fd/'calibrated_test.csv.gz',usecols=['subject_id']).subject_id.unique()
        for subject in test:
            if subject in fold_lookup:raise RuntimeError('Duplicate test subject across LOSO folds')
            fold_lookup[str(subject)]=fd
    folder=out/'cp14_matched_event_inference';folder.mkdir(parents=True,exist_ok=True)
    pairs=pairs[(pairs.matched.astype(str).str.lower().isin(['true','1'])) & pairs.threshold.isin(thresholds)].copy()
    if max_pairs is not None:pairs=pairs.head(max_pairs)
    pairs=pairs.reset_index(drop=True)
    baseline={(str(row.recording_id),round(float(row.start_sample)/fs,5)):row for row in events.itertuples(index=False)}
    record_rows=[];fail=[]
    for (rid,th),group in pairs.groupby(['recording_id','threshold'],sort=True):
        filename=folder/'parts'/f'{rid}_t{float(th):.3f}.csv.gz'
        cached=_cached_csv(filename)
        if cached is not None:
            record_rows.append(cached);continue
        evs=[]
        for j,r in group.iterrows():
            key=(str(rid),round(float(r.nominal_start_sec),5))
            nominal=baseline.get(key)
            if nominal is None:
                # tolerate CSV float rounding to sampling-time grid, but never an arbitrary nearest event.
                relevant=[e for e in events.itertuples(index=False) if str(e.recording_id)==str(rid) and abs(float(e.start_sample)/fs-float(r.nominal_start_sec))<=0.51/fs]
                if len(relevant)!=1:
                    fail.append(dict(recording_id=rid,threshold=th,reason='nominal_event_not_unique',start_sec=r.start_sec));continue
                nominal=relevant[0]
            event=pd.Series({'event_id':f'{nominal.event_id}__th{float(th):.3f}', 'recording_id':rid,
                 'subject_id':nominal.subject_id,'species':nominal.species,'path':nominal.path,
                 'start_sample':int(round(float(r.start_sec)*fs))})
            evs.append((event,str(nominal.event_id),float(r.onset_shift_sec)))
        if not evs:continue
        source=Path(str(evs[0][0]['path']))
        if not source.exists():
            fail.append(dict(recording_id=rid,threshold=th,reason='recording_not_found',start_sec=np.nan));continue
        amp,phase=read_signal_csv(source)
        z=preprocess_complex(complex_from_amplitude_phase(amp,phase),fs,cfg['signal'])
        output=[]
        for ev,nom_id,shift in evs:
            fd=fold_lookup.get(str(ev.subject_id))
            if fd is None:raise RuntimeError('No outer model for matched subject')
            feat=pd.DataFrame(extract_event_feature_rows(ev,z,fs,cfg['features']))
            if len(feat)!=len(ds):
                fail.append(dict(recording_id=rid,threshold=th,reason='incomplete_prefix',start_sec=float(ev.start_sample)/fs));continue
            temps=json.loads((fd/'temperatures.json').read_text())
            preds=[]
            for duration in ds:
                model_path=fd/'models'/f't{duration:.1f}.joblib'
                if not model_path.exists():raise FileNotFoundError(f'CP4 model missing: {model_path}')
                model=load_model_cached(str(model_path))
                sub=feat[np.isclose(feat.duration_sec,duration)].copy()
                x=sub.reindex(columns=model.feature_names).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(dtype=float)
                probs=model.predict_proba(x)
                t=temps[f'{duration:.1f}'];temperature=float(t['species'] if isinstance(t,dict) else t)
                cal=apply_temperature(probs,temperature)
                if not np.array_equal(probs.argmax(axis=1),cal.argmax(axis=1)):
                    raise RuntimeError('Calibration changed argmax')
                cols=['event_id','recording_id','subject_id','species','duration_sec','harmonic_ratio','harmonic_energy_ratio','sbr_db','f0_delta_hz','stability_available']
                p=sub[cols].copy()
                for k,s in enumerate(SPECIES):p[f'p_{s}']=cal[:,k]
                preds.append(p)
            pred=pd.concat(preds,ignore_index=True)
            fold=int(fd.name[-3:]);pol=_policy_for_fold(root,fold,eps)
            dec=apply_policy(pred,ds,pol)
            dec['nominal_event_id']=nom_id;dec['threshold']=th;dec['onset_shift_sec']=shift;dec['fold']=fold
            output.append(dec)
        if output:
            tab=pd.concat(output,ignore_index=True)
            _safe_write(tab,filename);record_rows.append(tab)
    _safe_write(pd.DataFrame(fail,columns=['recording_id','threshold','reason','start_sec']),folder/'cp14_missing_or_excluded.csv')
    if not record_rows:
        atomic_write_json(folder/'status.json',dict(complete=False,processed=0,reason='No matched windows inferred; check dataset paths'))
        return
    a=pd.concat(record_rows,ignore_index=True)
    if a.duplicated(['nominal_event_id','threshold']).any():raise RuntimeError('Duplicate matched nominal IDs')
    _safe_write(a,folder/'cp14_shifted_decisions.csv.gz')
    # derive *nominal decisions* with exactly the same frozen outer-fold policy
    base=[]
    for fd in folds(root):
        sub=pd.read_csv(fd/'calibrated_test.csv.gz');f=int(fd.name[-3:])
        dec=apply_policy(sub,durations(sub),_policy_for_fold(root,f,eps))
        dec['fold']=f;base.append(dec)
    base=pd.concat(base,ignore_index=True)
    merged=a.merge(base[['event_id','pred_species','stop_duration_sec']].rename(columns={'event_id':'nominal_event_id','pred_species':'nominal_pred_species','stop_duration_sec':'nominal_stop_duration_sec'}),on='nominal_event_id',how='left',validate='many_to_one')
    if merged.nominal_pred_species.isna().any():raise RuntimeError('Missing nominal decision for matched pair')
    merged['changed_prediction']=merged.pred_species!=merged.nominal_pred_species
    merged['delta_stop_sec']=merged.stop_duration_sec-merged.nominal_stop_duration_sec
    # Pairwise output written after taxonomic gaps have been computed below.
    result=[]
    # Each event uses the loss matrix stored for its independent outer fold.
    matrices={}
    for fd in folds(root):
        f=int(fd.name[-3:]); matrices[f]=_load_fold(root,fd)[2]
    lut={name:i for i,name in enumerate(SPECIES)}
    gaps=[]; nominal_gaps=[]
    for row in merged.itertuples(index=False):
        L=matrices[int(row.fold)]
        orig=base.loc[base.event_id==row.nominal_event_id].iloc[0]
        species_idx=lut[row.species]; pred_idx=lut[row.pred_species]; full_idx=lut[orig.full_pred_species]
        gaps.append(float(L[species_idx,pred_idx]-L[species_idx,full_idx]))
        nominal_gaps.append(float(L[species_idx,lut[row.nominal_pred_species]]-L[species_idx,full_idx]))
    merged['shifted_taxonomic_gap']=gaps
    merged['nominal_taxonomic_gap']=nominal_gaps
    result=[]
    for th,g in merged.groupby('threshold'):
        result.append(dict(threshold=th,matched_evaluated=len(g),subjects=g.subject_id.nunique(),
           shifted_accuracy=accuracy_score(g.species,g.pred_species),nominal_matched_accuracy=accuracy_score(g.species,g.nominal_pred_species),
           shifted_macro_f1=f1_score(g.species,g.pred_species,labels=SPECIES,average='macro',zero_division=0),
           nominal_matched_macro_f1=f1_score(g.species,g.nominal_pred_species,labels=SPECIES,average='macro',zero_division=0),
           changed_prediction_fraction=g.changed_prediction.mean(),mean_delta_stop_sec=g.delta_stop_sec.mean(),
           shifted_mean_stop_sec=g.stop_duration_sec.mean(),nominal_matched_mean_stop_sec=g.nominal_stop_duration_sec.mean(),
           shifted_subject_mean_gap=g.groupby('subject_id').shifted_taxonomic_gap.mean().mean(),
           nominal_matched_subject_mean_gap=g.groupby('subject_id').nominal_taxonomic_gap.mean().mean(),
           median_abs_onset_shift_sec=g.onset_shift_sec.abs().median()))
    _safe_write(pd.DataFrame(result),folder/'cp14_matched_metrics.csv')
    _safe_write(merged[['nominal_event_id','event_id','fold','threshold','species','subject_id','onset_shift_sec','pred_species','nominal_pred_species','stop_duration_sec','nominal_stop_duration_sec','changed_prediction','delta_stop_sec','shifted_taxonomic_gap','nominal_taxonomic_gap']],folder/'cp14_paired_comparison.csv.gz')
    atomic_write_json(folder/'status.json',dict(complete=max_pairs is None and len(fail)==0 and len(a)==len(pairs),
       processed=len(a),matched_requested=len(pairs),excluded=len(fail),
       scope='matched events only; not whole-detector population, no detection ground truth',
       warning='A complete end-to-end sensitivity claim requires accounting for unmatched/dropped/new events.'))

def verify(root,out,eps):
    rows=[]
    for stage,files in {'cp12_metric_audit':['cp12_pooled_recomputed.csv','cp12_metric_reconciliation.csv','status.json'],
      'cp13_training_only_simplification':['cp13_fold_choices.csv','cp13_pooled_metrics.csv','status.json'],
      'cp14_matched_event_inference':['cp14_matched_metrics.csv','cp14_paired_comparison.csv.gz','status.json']}.items():
        base=out/stage
        status=json.loads((base/'status.json').read_text()) if (base/'status.json').exists() else {}
        for fn in files:rows.append(dict(stage=stage,file=fn,exists=(base/fn).is_file(),bytes=(base/fn).stat().st_size if (base/fn).exists() else 0))
        rows.append(dict(stage=stage,file='STAGE_COMPLETE',exists=bool(status.get('complete',False)),bytes=0))
    table=pd.DataFrame(rows);_safe_write(table,out/'v62_output_checklist.csv')
    complete=bool(table.exists.all())
    cp12=json.loads((out/'cp12_metric_audit'/'status.json').read_text()) if (out/'cp12_metric_audit'/'status.json').exists() else {}
    atomic_write_json(out/'v62_verification.json',dict(files_complete=complete,metric_reconciled=cp12.get('reconciled',False),
      manuscript_ready=complete and cp12.get('reconciled',False),warning='Formal reviewer satisfaction still requires human checking of manuscript claims and matched-event coverage.'))
    print(f'files_complete={complete}; metric_reconciled={cp12.get("reconciled",False)}')

def main():
    p=argparse.ArgumentParser(description='QTRC v6.2 reviewer validation')
    p.add_argument('stage',choices=['audit','simplify','onset','verify','all'])
    p.add_argument('--run-dir',required=True);p.add_argument('--epsilon',type=float,default=.05)
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--bootstrap-repeats',type=int,default=1000)
    p.add_argument('--max-pairs',type=int,default=None,help='Smoke test only; not manuscript results')
    a=p.parse_args();root=Path(a.run_dir).resolve();out=root/'reviewer_extension_v62'
    if a.stage in ['audit','all']:audit(root,out,a.epsilon)
    if a.stage in ['simplify','all']:simplify(root,out,a.epsilon,a.seed,a.bootstrap_repeats)
    if a.stage in ['onset','all']:onset(root,out,a.epsilon,max_pairs=a.max_pairs)
    if a.stage in ['verify','all']:verify(root,out,a.epsilon)
if __name__=='__main__':main()
