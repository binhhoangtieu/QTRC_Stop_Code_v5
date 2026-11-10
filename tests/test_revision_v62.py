import numpy as np
import pandas as pd
from qtrc_stop.revision_v62 import _loss, _ucb, DROPS
from qtrc_stop.taxonomy import SPECIES

def test_gap_zero_same_full():
    a=pd.DataFrame({'species':[SPECIES[0],SPECIES[1]],'pred_species':[SPECIES[1],SPECIES[1]],'full_pred_species':[SPECIES[1],SPECIES[1]]})
    assert np.all(_loss(a,np.ones((5,5)))==0)

def test_bootstrap_ucb_constant():
    assert abs(_ucb(np.array([0.01,0.01,0.01]),np.random.default_rng(3),100)-0.01)<1e-12

def test_no_confidence_deletion_candidate():
    assert 'confidence' not in DROPS
