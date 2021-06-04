import os
import numpy as np
import matplotlib.pyplot as plt

from src.codec.pcm import PulseCodeModulation
from src.codec.bin import BinaryVectorModulation
from src.codec.grid import GridCellModulation
from src.codec.pdm import PulseDensityModulation
from src.net.slayer import SLAYERAudioPipeline

def interactive_legend(ax=None):
    """FROM: https://stackoverflow.com/questions/31410043/hiding-lines-after-showing-a-pyplot-figure"""
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.get_legend())

class InteractiveLegend(object):
    """FROM: https://stackoverflow.com/questions/31410043/hiding-lines-after-showing-a-pyplot-figure"""
    def __init__(self, legend):
        self.legend = legend
        self.fig = legend.axes.figure

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:

            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def show(self):
        plt.show()



def slayer_benchmark():
    errors = {}
    vsizes = {}
    audios = {}
    for name, codec, model, layer_sizes in zip(titles, codecs, models, layers):
        print('---', name, '---')
        error_list = []
        audio_list = []

        # BUILD FEATURE VECTORS
        if codec == pdm:
            input_audio.oversample(OSR=oversampling)
            input_wave_os = input_audio.get_waveform(wave_format='Channel-Amplitude', mono=mono)
            input_vectors = np.array(codec.value2vector(input_wave_os))
            use_batches = False
        else:
            input_vectors = np.array(codec.value2vector(input_wave))
            use_batches = False
        desir_vectors = input_vectors

        # RUN NETWORK
        network_audio = SLAYERAudioPipeline(config_path=configpath,
                                            input_size=codec.get_vector_size(),
                                            output_size=codec.get_vector_size(),
                                            samplerate=input_audio.samplerate,
                                            mono=mono,
                                            layer_sizes=layer_sizes)

        outpath_model = f'{outpath_netwk}{model}-{modelversion}.pt'
        if not os.path.exists(outpath_model):
            print(f'\n>>> Train New Model ({model}-{modelversion}.pt)')
            network_audio.visualize_training(logpath=f'{outpath_netwk_plt}{model}-{modelversion}', subtitle=name)
            network_audio.train(input_vectors, desir_vectors, n_epochs, checkpoint, outpath_model)
        else:
            print(f'\n>>> Load Model ({model}-{modelversion}.pt)')
            network_audio.load(outpath_model)

        # OUTPUT
        output_vectors = network_audio.test(input_vectors, desir_vectors, logpath=outpath_audio_plt, use_batches=use_batches)
        output_wave = np.array(codec.vector2value(output_vectors))
        output_audio = PulseCodeModulation().from_array(output_wave, input_audio.samplerate, array_format='Channel-Amplitude')
        output_audio.to_file(outpath_audio, f'synthout-{model}')
        output_audio.play()
        print(f">>> Audio saved to '{outpath_audio}synthout-{model}.wav'")

        print()

def visualize_benchmark():
    fig, ax = plt.subplots()

    for name, codec, model, layer_sizes in zip(titles, codecs, models, layers):
        wave_path = f'{outpath_audio}synthout-{model}.wav'
        output_audio = PulseCodeModulation().from_file(wave_path, dtype=f'int{bit_depth}')
        ax.plot(output_audio.waveform, label=name)

    ax.plot(input_wave.flatten(), label='Original', c='black', linewidth=0.5)
    ax.legend(loc='upper right', ncol=1) # borderaxespad=0, bbox_to_anchor=(1.05, 1)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')

    #fig.subplots_adjust(right=0.55)
    fig.suptitle('Right-click to hide all. Middle-click to show all',
                 va='top', size='small')

    leg = interactive_legend()
    plt.title('SLAYER Benchmark')
    plt.show()

def log_results():
    fig, axs = plt.subplots(len(titles)+1, 1, sharex=True, sharey=True, figsize=(outputsize[0]/scaling, outputsize[1]/scaling), dpi=scaling, constrained_layout=True)
    original_wave = input_audio.get_waveform(mono=True).flatten() / (2**bit_depth/2)

    axs[0].plot(original_wave, label='Input Signal (Spoken Language)', linewidth=0.5)
    axs[0].set_ylabel('Amplitude')
    #axs[0].set_title('Input Signal (Spoken Language)')
    axs[0].text(.5, .9, 'Input Signal (Spoken Language)', horizontalalignment='center', transform=axs[0].transAxes, fontsize=16)
    #axs[0].legend(loc='upper left', handletextpad=-0.05, handlelength=0)

    print('Errors (min, mean, max)')
    errors = {}
    for ax, name, codec, model, layer_sizes in zip(axs[1:], titles, codecs, models, layers):
        wave_path = f'{outpath_audio}synthout-{model}.wav'
        output_audio = PulseCodeModulation().from_file(wave_path, dtype=f'int{bit_depth}')
        slayer_wave = output_audio.waveform / (2**bit_depth/2)
        ax.plot(slayer_wave, label=name, linewidth=0.5)
        ax.set_ylabel('Amplitude')
        #ax.set_title(name)
        ax.text(.5, .9, name, horizontalalignment='center', transform=ax.transAxes, fontsize=16)
        #ax.legend(loc='upper left', handletextpad=-0.05, handlelength=0)

        # EVALUATION METRICS
        error = np.abs(original_wave - slayer_wave)
        errors[name] = (round(np.min(error),4), round(np.mean(error),4), round(np.max(error),4))
        print(f'{name:>25}: {errors[name]}')

    axs[-1].set_xlabel('Sample')
    fig.suptitle('SLAYER Output', fontsize=24)
    plt.savefig(f'{outpath_audio_plt}slayer_benchmark.png', dpi=scaling * outputdpi)
    plt.show()


if __name__ == '__main__':
    # GENERAL
    outputpath = '../../save/'

    # VISUALIZE
    outputsize = (900, 1080)
    scaling = 50
    outputdpi = 10

    # AUDIO
    audiofile = 'speech1'
    start = 8000
    end = 50000
    bit_depth = 16
    oversampling = 16

    # MODEL
    configpath = '../../config/net_slayer.yaml'
    modelversion = '002'
    n_epochs = 3000
    checkpoint = 100

    # CREATE FOLDERS
    outpath_audio = outputpath + f'audio/{audiofile}/'
    outpath_audio_plt = outputpath + f'audio/{audiofile}/plot/'
    outpath_netwk = outputpath + f'net/slayer/{modelversion}/'
    outpath_netwk_plt = outputpath + f'net/slayer/{modelversion}/plot/'
    for path in (outpath_audio, outpath_audio_plt, outpath_netwk, outpath_netwk_plt):
        if not os.path.exists(path):
            os.makedirs(path)
    audiopath = f'../../res/audio/input/{audiofile}.wav'

    # GET INPUT WAVE
    mono = True
    input_audio = PulseCodeModulation().from_file(audiopath, dtype=f'int{bit_depth}', start=start, end=end)
    #input_audio.play()
    input_audio.to_file(outpath_audio, 'original')

    input_wave = input_audio.get_waveform(wave_format='Channel-Amplitude', mono=mono)
    desire = input_wave.flatten()

    # BUILD ENCODERS
    bin = BinaryVectorModulation(bit_depth)
    wbn = BinaryVectorModulation(bit_depth, weighted=True)
    gcm = GridCellModulation(2 ** bit_depth, seed=0).new_modules(module_count=80, cells_per_module=10)
    pdm = PulseDensityModulation(order=6, OSR=oversampling)

    # ITERATE THROUGH ALL ENCODERS
    codecs = [bin, wbn, gcm, pdm]
    titles = ["Two's Complement", "Weighted Two's Complement", 'Grid Cell Modules', 'Delta Sigma Modulation']
    models = ['d2comp', 'w2comp', 'grid', 'dsm']
    layers = [[128], [128], [1024], [128]]

    # codecs = [pdm, wbn]
    # titles = ['Delta Sigma Modulation', 'asd']
    # models = ['dsm', 'w2comp']
    # layers = [[128], [128]]

    # RUN
    slayer_benchmark()
    #visualize_benchmark()
    #log_results()