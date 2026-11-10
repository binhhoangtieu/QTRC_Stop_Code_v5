import numpy as np
from qtrc_stop.taxonomy import SPECIES, taxonomic_loss_matrix


def test_taxonomic_loss_properties():
    counts = {s: 5 for s in SPECIES}
    mat = taxonomic_loss_matrix(counts)
    assert mat.shape == (5, 5)
    assert np.allclose(np.diag(mat), 0.0)
    assert np.all((mat >= 0) & (mat <= 1))
