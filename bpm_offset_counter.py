import array
import math
import wave
import numpy as np
import pywt
from scipy import signal
import librosa


def read_wav(filename):
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return None, None

    nsamps = wf.getnframes()
    if nsamps <= 0:
        return None, None

    fs = wf.getframerate()
    if fs <= 0:
        return None, None

    samps = list(array.array("i", wf.readframes(nsamps)))
    return samps, fs


def peak_detect(data):
    max_val = np.amax(np.abs(data))
    peak_ndx = np.where(data == max_val)
    if len(peak_ndx[0]) == 0:
        peak_ndx = np.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    # Use 4-level wavelet decomposition
    levels = 4
    max_decimation = 2 ** (levels - 1)
    cD_sum = None
    cD_minlen = None

    for loop in range(0, levels):
        if loop == 0:
            cA, cD = pywt.dwt(data, "db4")
            cD_minlen = len(cD) // max_decimation + 1
            cD_sum = np.zeros(int(cD_minlen))
        else:
            cA, cD = pywt.dwt(cA, "db4")
        # Apply a simple high-pass filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)
        decimation = 2 ** (levels - loop - 1)
        cD = np.abs(cD[::decimation])
        cD = cD - np.mean(cD)
        cD_sum = cD[0:int(cD_minlen)] + cD_sum

    # If cA is all zeros, no audio data available.
    if all(b == 0.0 for b in cA):
        return None, None

    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = np.abs(cA)
    cA = cA - np.mean(cA)
    cD_sum = cA[0:int(cD_minlen)] + cD_sum
    correl = np.correlate(cD_sum, cD_sum, "full")
    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    # If multiple peaks are detected, skip this segment.
    if len(peak_ndx[0]) > 1:
        return None, correl
    peak_ndx_adjusted = peak_ndx[0][0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    return bpm, correl


def wavelet_bpm_detector(data, fs, window_sec=8.0):
    """
    Splits the audio into windows and computes BPM for each using the wavelet method.
    Returns the median BPM over windows.
    """
    nsamps = len(data)
    window_samps = int(window_sec * fs)
    max_window_ndx = nsamps // window_samps
    bpms = []
    for window_ndx in range(max_window_ndx):
        start = window_ndx * window_samps
        segment = data[start: start + window_samps]
        bpm, _ = bpm_detector(segment, fs)
        if bpm is not None:
            bpms.append(bpm)
    if len(bpms) == 0:
        return None
    return np.median(bpms)


# === Spectral Flux Onset Detection & Offset Estimation (Simplified) ===

def compute_spectral_flux(y, sr, n_fft=1024, hop_length=256):
    """
    Compute spectral flux from the STFT of the signal.
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    flux = np.zeros(mag.shape[1])
    for n in range(1, mag.shape[1]):
        diff = mag[:, n] - mag[:, n - 1]
        flux[n] = np.sum(np.maximum(0, diff))
    return flux, hop_length


def peak_picking(flux, hop_length, sr, window_size=7, alpha=0.1):
    """
    Temporal peak picking to extract onset frame indices.
    """
    onsets = []
    L = len(flux)
    for n in range(5, L - 1):
        local_window = flux[n - 5: n + 2]
        threshold = np.median(local_window) + alpha * np.mean(local_window)
        if flux[n] == np.max(local_window) and flux[n] > threshold:
            onsets.append(n)
    onset_samples = np.array(onsets) * hop_length
    return onset_samples


def compute_offset_from_onsets(onset_samples, sr, interval, hamming_size=20):
    """
    Given onset sample positions, build a histogram modulo the beat interval and
    return the bin with maximum count as the offset (in seconds).
    """
    H = np.zeros(interval)
    for p in onset_samples:
        H[int(p % interval)] += 1
    hamming_win = np.hamming(hamming_size)
    H_smooth = np.convolve(H, hamming_win, mode="same")
    best_offset_bin = np.argmax(H_smooth)
    return best_offset_bin / sr


def main(filename):
    # Read the audio file using the wave module.
    print(filename)
    samps, fs = read_wav(filename)
    if samps is None:
        print("Failed to read audio.")
        return
    duration_sec = len(samps) / fs
    print(f"Loaded audio with sampling rate {fs} Hz; duration = {duration_sec:.2f} sec.")

    # Estimate BPM using the robust wavelet-based method.
    bpm = wavelet_bpm_detector(samps, fs, window_sec=8.0)
    if bpm is None:
        print("BPM detection failed.")
        return
    if bpm < 50:
        bpm = bpm * 4
    print(f"Estimated BPM: {bpm:.2f}")

    # Convert integer samples to a float array (normalized) for spectral processing.
    y = np.array(samps, dtype=np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    else:
        print("Silent audio.")
        return

    # Onset detection using spectral flux.
    flux, hop_length = compute_spectral_flux(y, fs, n_fft=1024, hop_length=256)
    onset_samples = peak_picking(flux, hop_length, fs, window_size=7, alpha=0.1)

    # Compute offset from onsets.
    # The beat interval (in samples) corresponding to the detected BPM.
    interval = int(fs * 60 / bpm)
    offset = compute_offset_from_onsets(onset_samples, fs, interval, hamming_size=20)
    print(f"Estimated offset: -{offset:.3f} sec")


if __name__ == "__main__":
    main("samples/aitai.wav")
    print("----------------------------")
    main("samples/monitoring.wav")
    print("----------------------------")
    main("samples/neppa.wav")
    print("----------------------------")
    main("samples/enma.wav")
    print("----------------------------")
    main("samples/meltdown.wav")
    print("----------------------------")
    main("samples/mesmerizer.wav")
    print("----------------------------")
    main("samples/propose.wav")
