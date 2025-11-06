import os
from pathlib import Path
import numpy as np
from scipy.signal import welch, hilbert
from scipy.io import wavfile
def whiten(strain, fs, seglen=4, overlap=2, eps=1e-12):
    x = np.asarray(strain, dtype=float)
    x = x - np.mean(x)             # 先去均值
    n = len(x)

    # nperseg / noverlap 保护，避免 noverlap >= nperseg
    nperseg = min(max(int(seglen*fs), 256), n)
    noverlap = min(max(int(overlap*fs), 0), nperseg-1)

    # Welch PSD
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 插值到 rfft 频率，并防止除零
    f_fft = np.fft.rfftfreq(n, d=1.0/fs)
    psd_i = np.interp(f_fft, freqs, psd)
    psd_i = np.maximum(psd_i, eps)

    # 关键：对 welch 的正确归一化 —— 除以 sqrt(psd) * sqrt(2*fs)
    htilde = np.fft.rfft(x)
    htilde[0] = 0.0                # 屏蔽 DC，避免 0Hz 被放大
    white_htilde = htilde / (np.sqrt(psd_i) * np.sqrt(2.0*fs))
    white = np.fft.irfft(white_htilde, n=n)

    white -= np.mean(white)        # 数值稳态再去均值
    return white
def reqshift(data, fshift, fs):
    """解析信号乘相位 e^{j2πf t} 做频移，返回实部。"""
    t = np.arange(len(data)) / float(fs)
    analytic = hilbert(data)
    shifted = analytic * np.exp(2j * np.pi * fshift * t)
    return np.real(shifted)

def write_wavfile(filename, fs, data):
    """写 16-bit PCM WAV（自动建目录；float 自动规一化）。"""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(data)
    if np.issubdtype(x.dtype, np.floating):
        maxv = np.max(np.abs(x)) or 1.0
        x = (x / maxv * 32767.0).astype(np.int16)
    elif x.dtype != np.int16:
        x = x.astype(np.int16)

    wavfile.write(filename, int(fs), x)
    return str(filename)
