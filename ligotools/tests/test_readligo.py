import glob, numpy as np
from ligotools.readligo import loaddata

def _pick(pattern):
    files = sorted(glob.glob(pattern))
    assert files, f"not found：{pattern}"
    return files[0]

def _basic_checks(fn, ifo):
    strain, time, dq = loaddata(fn, ifo)

    assert isinstance(dq, dict)
    assert any(k.upper().startswith("BURST") for k in dq.keys())

    assert len(strain) == len(time) > 0
    assert np.isfinite(strain).all()
    assert np.all(np.diff(time) > 0)

    fs = 1.0 / np.median(np.diff(time))
    assert 4090 <= fs <= 4102

    duration = time[-1] - time[0]
    assert 31.0 <= duration <= 33.0

def test_loaddata_H1():
    fn = _pick("data/H-*-32.hdf5")
    _basic_checks(fn, "H1")

def test_loaddata_L1():
    fn = _pick("data/L-*-32.hdf5")
    _basic_checks(fn, "L1")
