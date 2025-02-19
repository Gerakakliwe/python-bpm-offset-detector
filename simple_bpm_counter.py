import array
import math
import wave
import numpy
import pywt
from scipy import signal


def read_wav(filename):
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None


def peak_detect(data):
    max_val = numpy.amax(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    cA = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        cD = signal.lfilter([0.01], [1 - 0.99], cD)
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)

        cD_sum = cD[0: math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0: math.floor(cD_minlen)] + cD_sum
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        return no_audio_data()
    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    return bpm, correl


def main(audio_file):
    filename = audio_file
    window = 8.0
    samps, fs = read_wav(filename)
    n = 0
    nsamps = len(samps)
    window_samps = int(window * fs)
    samps_ndx = 0
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = numpy.zeros(max_window_ndx)
    for window_ndx in range(0, max_window_ndx):
        data = samps[samps_ndx: samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        if bpm < 50:
            bpm = bpm * 4
        bpms[window_ndx] = bpm
        samps_ndx = samps_ndx + window_samps
        n = n + 1

    bpm = numpy.median(bpms)
    print(f"{audio_file}  Estimated Beats Per Minute:", bpm)


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
