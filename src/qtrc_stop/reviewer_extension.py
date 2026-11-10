"""Reviewer-specific CP9/CP10 on frozen calibrated LOSO predictions.
No classifier refit, no mixing of independent outer tests. Atomic per-task artifacts.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from .checkpoint import atomic_dataframe_csv, atomic_write_json
from .stopping import (Policy, apply_policy, prepare_policy_data, _candidate_indices,
                       _decisions_from_data, select_policy_split_certified, evaluate_policy_grid_data, subset_policy_data)
from .taxonomy import SPECIES, taxonomic_loss_matrix


def _load_fold(root: Path, fold_dir: Path):
    a=fold_dir/'calibrated_oof.csv.gz'; b=fold_dir/'calibrated_test.csv.gz'
    if not a.exists() or not b.exists():
        raise FileNotFoundError(f'Missing CP5 calibrated artifacts: {a}, {b}')
    loss_file=fold_dir/'loss_matrix.json'
    if not loss_file.exists():
        raise FileNotFoundError(f'Missing fold-specific training-only loss matrix: {loss_file}')
    return pd.read_csv(a),pd.read_csv(b),np.array(json.loads(loss_file.read_text())['matrix'],float)


def _metrics(dec, loss):
    true=dec.species.astype(str).to_numpy(); pred=dec.pred_species.astype(str).to_numpy()
    full=dec.full_pred_species.astype(str).to_numpy()
    lut={s:i for i,s in enumerate(SPECIES)}
    gaps=np.array([loss[lut[a],lut[b]]-loss[lut[a],lut[c]] for a,b,c in zip(true,pred,full)])
    subject_gaps=pd.DataFrame({'subject_id':dec.subject_id,'gap':gaps}).groupby('subject_id').gap.mean()
    return {'events':len(dec),'subjects':dec.subject_id.nunique(),
            'accuracy':accuracy_score(true,pred),'macro_f1':f1_score(true,pred,labels=SPECIES,average='macro',zero_division=0),
            'mean_stop_sec':float(dec.stop_duration_sec.mean()),'at_full_fraction':float(np.mean(np.isclose(dec.stop_duration_sec,2.0))),
            'subject_mean_gap':float(subject_gaps.mean()),'event_mean_gap':float(gaps.mean())}


def _decisions_without_gate(data, policy, drop):
    """One-gate deletion with every other frozen policy parameter unchanged."""
    conf=data['confidence']; q=data['quality']; n,t=conf.shape
    masks={
      'confidence': conf>=policy.confidence,
      'posterior_stability': data['js']<=policy.js_max,
      'prediction_persistence':data['persistence'].get(policy.persistence, np.ones((n,t),bool)),
      'sbr':q['sbr_db']>=policy.sbr_min_db,
      'f0_change':q['f0_delta_hz']<=policy.f0_delta_max_hz,
      'harmonic_ratio':q['harmonic_ratio']>=policy.harmonic_ratio_min,
      'harmonic_energy':q['harmonic_energy_ratio']>=policy.harmonic_energy_min,
      'stability_availability':(q['stability_available']>=0.5) if policy.require_stability_available else np.ones((n,t),bool),
    }
    eligible=np.ones((n,t),bool)
    for name,mask in masks.items():
        if name!=drop: eligible &= mask
    indices=np.where(eligible.any(axis=1),eligible.argmax(axis=1),t-1)
    dec=_decisions_from_data(data,policy,False)
    r=np.arange(n)
    dec['stop_duration_sec']=data['durations'][indices]
    dec['pred_species']=[SPECIES[j] for j in data['labels'][r,indices]]
    dec['confidence']=conf[r,indices]
    return dec


def _policy_for_fold(root,fold,epsilon):
    # CP5 policy saved as nested artifacts; verify exact identity from audit.
    matches=[root/'cp4_predictions'/f'outer_{fold:03d}'/'policies'/'qtrc_stop'/f'epsilon_{epsilon:.3f}.json']
    for f in matches:
        try:
            x=json.loads(f.read_text(encoding='utf-8'))
            if isinstance(x,dict):
                for obj in (x,x.get('policy',{})):
                    if isinstance(obj,dict) and 'confidence' in obj and 'epsilon' in obj and abs(float(obj['epsilon'])-epsilon)<1e-8:
                        return Policy(**{k:v for k,v in obj.items() if k in Policy.__dataclass_fields__})
        except (ValueError,TypeError,KeyError): pass
    raise FileNotFoundError(f'Frozen CP5 policy not located for fold {fold}, epsilon={epsilon}; inspect cp5_policies')


def run(root:Path, epsilon:float, seeds:int, cfg:dict, out:Path, do_cp9=True, do_cp10=True):
    folds=sorted((root/'cp4_predictions').glob('outer_*'))
    if len(folds)!=33:
        raise RuntimeError(f'Expected 33 LOSO folders, got {len(folds)}; no partial findings accepted')
    durations=list(map(float,cfg['features']['prefix_durations_sec']))
    manifest={'input_run_dir':str(root),'epsilon':epsilon,'seeds':seeds,'n_outer_folds':len(folds),
              'type':'frozen_predictions_reviewer_audit','bootstrap_not_distribution_free':True}
    out.mkdir(parents=True,exist_ok=True)
    atomic_write_json(out/'manifest.json',manifest)
    for folder in folds:
        fold=int(folder.name.split('_')[-1]); oof,test,loss=_load_fold(root,folder)
        assert set(oof.subject_id).isdisjoint(set(test.subject_id)),f'Leakage fold {fold}'
        if do_cp9:
            part=out/'cp9_split_stability'/f'outer_{fold:03d}'
            part.mkdir(parents=True,exist_ok=True)
            for s in range(seeds):
                dest=part/f'seed_{s:03d}.csv'
                if dest.exists() and dest.stat().st_size>100: continue
                seed=20261013+fold*100000+s
                # Selection ranking depends only on mean_gap and mean_time; the bootstrap UCB
                # column of the grid is not used by select_policy_split_certified.
                # Set grid bootstrap to 1 to avoid millions of unnecessary bootstrap draws;
                # certification still uses ORIGINAL cfg['policy'] with 2000 repeats.
                from .stopping import prepare_policy_data, split_calibration_subjects
                selection,checking=split_calibration_subjects(oof,seed,float(cfg['policy'].get('certification_fraction',0.5)))
                fast_cfg=dict(cfg['policy'],bootstrap_repeats=1)
                sel_data=subset_policy_data(prepare_policy_data(oof,durations),selection)
                grid_fast=evaluate_policy_grid_data(sel_data,loss,fast_cfg,seed)
                policy,grid,audit=select_policy_split_certified(oof,durations,epsilon,loss,cfg['policy'],seed,
                                         selection_grid=grid_fast,selection_subjects=selection,certification_subjects=checking)
                dec=apply_policy(test,durations,policy)
                row=dict(fold=fold,seed=s,selection_subjects=len(audit['selection_subjects']),
                         checking_subjects=len(audit['certification_subjects']),
                         selection_subject_ids=';'.join(audit['selection_subjects']),
                         checking_subject_ids=';'.join(audit['certification_subjects']),
                         certified=policy.certification_passed,fallback=policy.fallback_full_duration,
                         certification_ucb=policy.calibration_ucb_gap,
                         confidence=policy.confidence,js_max=policy.js_max,
                         persistence=policy.persistence,sbr_min_db=policy.sbr_min_db,
                         f0_delta_max_hz=policy.f0_delta_max_hz,**_metrics(dec,loss))
                atomic_dataframe_csv(pd.DataFrame([row]),dest)
        if do_cp10:
            part=out/'cp10_ablation'/f'outer_{fold:03d}'
            part.mkdir(parents=True,exist_ok=True)
            dest=part/'component_ablation.csv'
            if dest.exists() and dest.stat().st_size>100: continue
            policy=_policy_for_fold(root,fold,epsilon)
            data=prepare_policy_data(test,durations)
            rows=[]
            for name in ['full','confidence','posterior_stability','prediction_persistence','sbr','f0_change','harmonic_ratio','harmonic_energy','stability_availability']:
                dec=_decisions_from_data(data,policy,False) if name=='full' else _decisions_without_gate(data,policy,name)
                rows.append(dict(fold=fold,variant=name,frozen_thresholds=True,**_metrics(dec,loss)))
            atomic_dataframe_csv(pd.DataFrame(rows),dest)
    if do_cp9:
        parts=sorted((out/'cp9_split_stability').glob('outer_*/seed_*.csv'))
        if len(parts)==len(folds)*seeds:
            a=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True)
            atomic_dataframe_csv(a,out/'cp9_split_stability'/'cp9_split_seed_fold_metrics.csv')
            ag=a.groupby('seed').agg(mean_stop_sec=('mean_stop_sec','mean'),macro_f1=('macro_f1','mean'),accuracy=('accuracy','mean'),subject_mean_gap=('subject_mean_gap','mean'),certification_rate=('certified','mean'),fallback_rate=('fallback','mean')).reset_index()
            atomic_dataframe_csv(ag,out/'cp9_split_stability'/'cp9_split_stability_summary.csv')
            variability=a.groupby('fold')[['confidence','js_max','persistence','sbr_min_db','f0_delta_max_hz']].nunique().reset_index()
            atomic_dataframe_csv(variability,out/'cp9_split_stability'/'cp9_policy_threshold_variability.csv')
    if do_cp10:
        parts=sorted((out/'cp10_ablation').glob('outer_*/component_ablation.csv'))
        if len(parts)==len(folds):
            a=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True)
            atomic_dataframe_csv(a,out/'cp10_ablation'/'cp10_component_ablation.csv')
            baseline=a[a.variant=='full'].set_index('fold')
            pairs=a[a.variant!='full'].copy()
            for metric in ['mean_stop_sec','accuracy','macro_f1','subject_mean_gap']:
                pairs[f'delta_{metric}']=pairs[metric]-pairs.fold.map(baseline[metric])
            atomic_dataframe_csv(pairs,out/'cp10_ablation'/'cp10_paired_fold_differences.csv')
    atomic_write_json(out/'completion.json',{'cp9_completed':not do_cp9 or (out/'cp9_split_stability'/'cp9_split_stability_summary.csv').exists(),
                                             'cp10_completed':not do_cp10 or (out/'cp10_ablation'/'cp10_component_ablation.csv').exists()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epsilon',type=float,default=0.05);p.add_argument('--seeds',type=int,default=30)
    p.add_argument('--only',choices=['cp9','cp10','both'],default='both')
    a=p.parse_args()
    from .config import load_config
    run(Path(a.run_dir),a.epsilon,a.seeds,load_config(a.config),Path(a.out),a.only in ('cp9','both'),a.only in ('cp10','both'))
if __name__=='__main__': main()
