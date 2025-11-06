import glob, numpy as np
from ligotools.readligo import loaddata

def _pick(pattern):
    files = sorted(glob.glob(pattern))
    assert files, f"没有找到：{pattern}"
    return files[0]

def _basic_checks(fn, ifo):
    strain, time, dq = loaddata(fn, ifo)

    # 第三个返回值应该是 DQ 字典（包含 BURST 类掩码等）
    assert isinstance(dq, dict)
    # 常见键之一：BURST_CAT1（不同数据包键名可能略有差异，因此用“包含 BURST”更稳）
    assert any(k.upper().startswith("BURST") for k in dq.keys())

    # 基本数值合法性
    assert len(strain) == len(time) > 0
    assert np.isfinite(strain).all()
    assert np.all(np.diff(time) > 0)

    # 采样率≈4096 Hz（留容差，避免不同包略差别）
    fs = 1.0 / np.median(np.diff(time))
    assert 4090 <= fs <= 4102

    # 时长≈32 s（留容差）
    duration = time[-1] - time[0]
    assert 31.0 <= duration <= 33.0

def test_loaddata_H1():
    fn = _pick("data/H-*-32.hdf5")
    _basic_checks(fn, "H1")

def test_loaddata_L1():
    fn = _pick("data/L-*-32.hdf5")
    _basic_checks(fn, "L1")
