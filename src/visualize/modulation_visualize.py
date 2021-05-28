import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random

from src.codec.pcm import PulseCodeModulation
from src.codec.pdm import PulseDensityModulation


class ModulationVisualizer(object):
    def __init__(self, title=None, number_of_plots=1):
        self.number_of_rows = number_of_plots
        self.fig, self.axis = plt.subplots(nrows=number_of_plots, ncols=1, sharex='all', sharey='all')
        self.fig.set_size_inches(15, 8, forward=True)
        self.fig.suptitle(title)

        if number_of_plots > 1:
            ax = None
            for ax in self.axis:
                pass
            else:
                ax.set_xlabel('Sample')
        else:
            self.axis.set_xlabel('Sample')

    def plot_pcm(self, signal: PulseCodeModulation, axidx=0, index=0, title=None, label='PCM Signal'):
        ax = self.__get_axis__(axidx)

        if signal.original_waveform.shape[0] != signal.waveform.shape[0]:
            orig_wave = signal.original_waveform[:,0]       # to mono
            over_wave = signal.waveform[:,0]                # to mono
            original_size = orig_wave.shape[0]
            oversamp_size = over_wave.shape[0]

            print(orig_wave.shape, over_wave.shape)

            # Compare original waveform to oversampled waveform
            over_samples = np.arange(oversamp_size) * (original_size / oversamp_size)
            orig_samples = np.arange(original_size)
            ax.plot(over_samples, over_wave, label=label + " (Oversampled)", linewidth=0.5)
            ax.scatter(orig_samples, orig_wave, label=label + " (Original)", s=1, c='red')

        else:
            # Just plot the waveform
            samples = np.arange(signal.waveform.shape[0])
            ax.plot(samples, signal.waveform, label=label, linewidth=0.5)


        ax.legend(loc=2)
        ax.set_title(title)
        #ax.set_ylim([-1, 1])
        ax.set_ylabel('Amplitude')

    def plot_pdm(self, stream: PulseDensityModulation, axidx, index=0, title=None, label='PDM Bitstream'):
        ax = self.__get_axis__(axidx)

        sample = np.arange(stream.bitstream.shape[0]) / stream.OSR
        ax.step(sample, 0.5*stream.bitstream-2*index-0.5, label=label, linewidth=0.5, c=self.__color_by_index__(index))

        ax.legend(loc=2)
        ax.set_title(title)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axes.yaxis.set_ticklabels([])
        ax.set_ylabel('Bit')

    def show(self):
        plt.show()

    def __get_axis__(self, index=0):
        if self.number_of_rows > 1:
            return self.axis[index]
        else:
            return self.axis

    @staticmethod
    def __color_by_index__(index=0, darkness=0.2, seed=3):
        random.seed(index+seed)
        return [darkness + (1-darkness)*random.random() for _ in range(3)]
