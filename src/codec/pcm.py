import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from deltasigma import sinc_decimate
import re
import matplotlib.pyplot as plt


class PulseCodeModulation(object):
    def __init__(self, samplerate=None, OSR=1, os_strategy=None):
        self.original_waveform = None
        self.float_waveform = None
        self.waveform = None
        self.samplerate = samplerate
        self.original_samplerate = samplerate
        self.librosa_waveform = None
        self.librosa_samplerate = None
        self.OSR = OSR
        self.strategy = os_strategy
        self.is_mono = False

    def from_file(self, path, dtype='float64', start=None, end=None):
        self.original_waveform, self.samplerate = sf.read(path, dtype=dtype)
        self.original_waveform = self.original_waveform[start:end]

        self.librosa_waveform, self.librosa_samplerate = librosa.load(path)
        osr_factor = self.librosa_samplerate / self.samplerate
        if start: start = int(start * osr_factor)
        if end: end = int(end * osr_factor)
        self.librosa_waveform = self.librosa_waveform[start:end]

        self.check_mono(self.original_waveform)
        self.oversample()
        return self

    def to_file(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        sf.write(f'{path}{filename}.wav', np.array(self.waveform, dtype=np.int16), self.samplerate)

    def from_array(self, wave, samplerate, array_format='Channel-Amplitude'):
        wave = np.array(wave)

        # ensure correct format
        reorder = tuple(int(pos) for pos in re.split(r'\D+', array_format.upper().replace('CHANNEL', '1').replace('AMPLITUDE', '0')))
        wave = np.moveaxis(wave, (0, 1), reorder)

        if wave.dtype.kind not in 'iu':     # make sure the type is int16
            wave = self.float2int(wave)

        self.original_waveform = wave
        self.samplerate = samplerate
        # TODO: self.librosa_waveform, self.librosa_samplerate = librosa.load(path)
        self.check_mono(self.original_waveform)
        self.original_samplerate = samplerate
        self.oversample()
        return self

    def from_pdm(self, reference):
        sinc_order = 16     # TODO
        self.original_waveform = sinc_decimate(reference.bitstream, sinc_order, reference.OSR)
        self.check_mono(self.original_waveform)
        self.oversample()
        return self

    def play(self, volume=10000, blocking=True, message=''):
        if len(self.original_waveform.shape) == 1:
            max_ampl = max(self.original_waveform)
        elif len(self.original_waveform.shape) == 2:
            max_ampl = max(self.original_waveform[:,0])
        else:
            max_ampl = 1

        if max_ampl > 1:
            volume = max_ampl / volume

        audio = np.array(self.original_waveform * volume, dtype=np.int16)     # Convert to wav format (16 bits)
        print(message)
        sd.play(audio, blocking=blocking, samplerate=self.original_samplerate)         # Playback

    def get_waveform(self, wave_format='Channel-Amplitude', mono=True):
        # ensure correct format
        reorder = tuple(int(pos) for pos in re.split(r'\D+', wave_format.upper().replace('CHANNEL', '1').replace('AMPLITUDE', '0')))
        wave = np.moveaxis(np.array(self.waveform), (0, 1), reorder)

        # convert to mono if wished
        is_mono = self.check_mono(wave)
        if not is_mono and mono:
            # stereo -> mono
            wave = [wave[0]]
            is_mono = True
        elif is_mono and not mono:
            # mono -> stereo
            wave = [wave[0], wave[0]]
            is_mono = False
        return np.array(wave)

    def check_mono(self, wave):
        try:
            if len(wave[0]) == 1:
                self.is_mono = True
            else:
                self.is_mono = False
        except TypeError:
            self.is_mono = True
        return self.is_mono

    def oversample(self, OSR=None, dtype='float64'):
        if not OSR:
            OSR = self.OSR
        else:
            self.OSR = OSR

        if OSR > 1:
            if self.strategy == 0:
                res_type = 'zero_order_hold'  # very fast
            elif self.strategy == 1:
                res_type = 'linear'  # fast
            elif self.strategy == 2:
                res_type = 'kaiser_fast'  # medium
            elif self.strategy == 3:
                res_type = 'kaiser_best'  # slow
            elif self.strategy == 4:
                res_type = 'polyphase'  # fast
            elif self.strategy == 5:
                res_type = 'sinc_best'  # slow
            elif self.strategy == 6:
                res_type = 'sinc_medium'  # medium
            elif self.strategy == 7:
                res_type = 'sinc_fastest'  # fast
            else:
                res_type = 'linear'
            osr_factor = self.samplerate // self.librosa_samplerate     # fix inconsistencies with samplerates
            oversampled_wave = librosa.resample(self.librosa_waveform, self.librosa_samplerate, self.librosa_samplerate * osr_factor * self.OSR, res_type=res_type)
            if dtype == 'int16':
                oversampled_wave = self.float2int(oversampled_wave)
            self.waveform = np.expand_dims(oversampled_wave, axis=0).T
            self.samplerate = self.librosa_samplerate
        else:
            self.waveform = self.original_waveform

    @staticmethod
    def int2float(wave, dtype='float32'):
        """
        Convert waveform from int16 to float32.
        From: https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
        """
        sig = np.asarray(wave)
        dtype = np.dtype(dtype)
        dtype_info = np.iinfo(sig.dtype)
        abs_max = 2 ** (dtype_info.bits - 1)
        offset = dtype_info.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max

    @staticmethod
    def float2int(wave, dtype='int16'):
        """
        Convert waveform from float32 to int16.
        From: https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
        """
        dtype = np.dtype(dtype)
        dtype_info = np.iinfo(dtype)
        abs_max = 2 ** (dtype_info.bits - 1)
        offset = dtype_info.min + abs_max
        return (wave * abs_max + offset).clip(dtype_info.min, dtype_info.max).astype(dtype)