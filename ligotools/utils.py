import os
from pathlib import Path
import numpy as np
from scipy.signal import welch, hilbert
from scipy.io import wavfile
def whiten(strain, fs, seglen=4, overlap=2, eps=1e-12):
    x = np.asarray(strain, dtype=float)
    x = x - np.mean(x)
    n = len(x)

    nperseg = min(max(int(seglen*fs), 256), n)
    noverlap = min(max(int(overlap*fs), 0), nperseg-1)

    # Welch PSD
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    f_fft = np.fft.rfftfreq(n, d=1.0/fs)
    psd_i = np.interp(f_fft, freqs, psd)
    psd_i = np.maximum(psd_i, eps)

    htilde = np.fft.rfft(x)
    htilde[0] = 0.0              
    white_htilde = htilde / (np.sqrt(psd_i) * np.sqrt(2.0*fs))
    white = np.fft.irfft(white_htilde, n=n)

    white -= np.mean(white)     
    return white
def reqshift(data, fshift, fs):
    """Multiply the analytic signal by its phase e^{j2πf t}, perform a frequency shift, and return to the real part."""
    t = np.arange(len(data)) / float(fs)
    analytic = hilbert(data)
    shifted = analytic * np.exp(2j * np.pi * fshift * t)
    return np.real(shifted)

def write_wavfile(filename, fs, data):
    """write 16-bit PCM WAV（Automatic directory creation; float for automatic normalization）。"""
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
