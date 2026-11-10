"""CP11 detector-only threshold sensitivity, NOT a classification robustness test."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .checkpoint import atomic_dataframe_csv,atomic_write_json
from .config import load_config
from .events import detect_events
from .io import read_signal_csv
from .signal import complex_from_amplitude_phase,preprocess_complex


def run(root:Path, cfg:dict, out:Path, thresholds:list[float]):
    inventory=root/'cp0_inventory'/'selected'/'recordings.csv'
    if not inventory.exists(): raise FileNotFoundError('CP0 selected recordings missing')
    rows=pd.read_csv(inventory)
    out.mkdir(parents=True,exist_ok=True)
    fs=float(cfg['dataset']['sampling_rate_hz'])
    nominal=float(cfg['events']['harmonic_ratio_threshold'])
    if nominal not in thresholds: thresholds=sorted(set(thresholds+[nominal]))
    parts=out/'recordings';parts.mkdir(exist_ok=True)
    for rec in rows.itertuples(index=False):
        dest=parts/f'{rec.recording_id}.csv'
        if dest.exists() and dest.stat().st_size>30: continue
        amp,phase=read_signal_csv(rec.path)
        z=preprocess_complex(complex_from_amplitude_phase(amp,phase),fs,cfg['signal'])
        records=[]
        for threshold in thresholds:
            options=dict(cfg['events'],harmonic_ratio_threshold=threshold)
            ev=detect_events(z,fs,options)
            for idx,e in enumerate(ev):
                records.append({'recording_id':rec.recording_id,'subject_id':rec.subject_id,
                                'species':rec.species,'threshold':threshold,'ordinal':idx,
                                'start_sample':e.start_sample,'start_sec':e.start_sample/fs})
        atomic_dataframe_csv(pd.DataFrame(records,columns=['recording_id','subject_id','species','threshold','ordinal','start_sample','start_sec']),dest)
    sample=[pd.read_csv(p) for p in sorted(parts.glob('*.csv'))]
    a=pd.concat(sample,ignore_index=True)
    atomic_dataframe_csv(a,out/'cp11_detector_event_inventory.csv')
    summ=[];matching=[]
    for threshold in thresholds:
        frame=a[np.isclose(a.threshold,threshold)]
        counts=frame.groupby('recording_id').size()
        summ.append({'threshold':threshold,'events':len(frame),'subjects_with_events':frame.subject_id.nunique(),
                     'recordings_with_events':len(counts),'mean_events_per_detected_recording':float(counts.mean()) if len(counts) else 0})
        if np.isclose(threshold,nominal):continue
        shifts=[]
        for rid,group in frame.groupby('recording_id'):
            base=a[(a.recording_id==rid)&np.isclose(a.threshold,nominal)]
            # nearest monotone one-to-one matching, maximum 0.5 sec; unmatched are explicit
            candidates=list(base.start_sec.to_numpy());used=set()
            for t in group.start_sec:
                viable=[(abs(t-b),j,b) for j,b in enumerate(candidates) if j not in used and abs(t-b)<=0.5]
                if viable:
                    d,j,b=min(viable);used.add(j);shifts.append(float(t-b))
                    matching.append({'recording_id':rid,'threshold':threshold,'start_sec':t,'nominal_start_sec':b,'onset_shift_sec':t-b,'matched':True})
                else:matching.append({'recording_id':rid,'threshold':threshold,'start_sec':t,'nominal_start_sec':np.nan,'onset_shift_sec':np.nan,'matched':False})
        summ[-1]['matched_events']=len(shifts)
        summ[-1]['median_abs_onset_shift_sec']=float(np.median(np.abs(shifts))) if shifts else np.nan
    atomic_dataframe_csv(pd.DataFrame(summ),out/'cp11_detector_threshold_sensitivity.csv')
    atomic_dataframe_csv(pd.DataFrame(matching),out/'cp11_event_matching.csv')
    atomic_write_json(out/'status.json',{'complete':True,'interpretation':'detector-only; no event-level ground truth; classification metrics require new properly matched predictions','nominal':nominal,'thresholds':thresholds})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--thresholds',nargs='+',type=float,default=[.30,.35,.40]);a=p.parse_args()
    run(Path(a.run_dir),load_config(a.config),Path(a.out),a.thresholds)
