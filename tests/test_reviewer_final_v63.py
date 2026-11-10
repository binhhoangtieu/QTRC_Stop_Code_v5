import numpy as np
import pandas as pd
from qtrc_stop.revision_v62 import _loss
from qtrc_stop.taxonomy import SPECIES

def test_gap_is_signed():
    L=np.zeros((len(SPECIES),len(SPECIES)))
    L[0,1]=1.
    df=pd.DataFrame({'species':[SPECIES[0]],'pred_species':[SPECIES[0]],'full_pred_species':[SPECIES[1]]})
    assert _loss(df,L)[0]==-1.

def test_loss_equal():
    L=np.eye(len(SPECIES))
    df=pd.DataFrame({'species':[SPECIES[0]],'pred_species':[SPECIES[1]],'full_pred_species':[SPECIES[1]]})
    assert _loss(df,L)[0]==0.
