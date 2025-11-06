import os, numpy as np
from pathlib import Path
from scipy.io import wavfile
from numpy.fft import rfft, rfftfreq

from ligotools.utils import whiten, reqshift, write_wavfile

def test_write_wavfile_roundtrip(tmp_path):
    fs = 4096
    t = np.arange(fs) / fs
    x = 0.8*np.sin(2*np.pi*440*t)  # 1 秒 440Hz
    out = write_wavfile(tmp_path/"test.wav", fs, x)
    assert os.path.exists(out)
    rfs, y = wavfile.read(out)
    assert rfs == fs
    assert y.dtype == np.int16
    assert len(y) == fs

def test_reqshift_frequency():
    fs = 4096
    N = 4096
    t = np.arange(N)/fs
    f0, df = 100.0, 50.0
    x = np.sin(2*np.pi*f0*t)
    y = reqshift(x, df, fs)
    # 看频谱峰值是否从 100Hz 变到约 150Hz
    freqs = rfftfreq(N, 1/fs)
    peak = freqs[np.argmax(np.abs(rfft(y)))]
    assert abs(peak - (f0+df)) < 2.0

def test_whiten_basic_stats():
    fs = 4096
    rng = np.random.default_rng(0)
    # 轻度着色噪声：一阶滤波叠加
    white = rng.standard_normal(fs*2)
    colored = np.convolve(white, [1, 0.9], mode="same")
    w = whiten(colored, fs)
    # 零均值、单位量级方差（给宽容差以避免平台差异）
    assert abs(np.mean(w)) < 0.2
    std = np.std(w)
    assert 0.5 < std < 2.0
