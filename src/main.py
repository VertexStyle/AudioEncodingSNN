"""
Marcel Rinder - Künstliche Intelligenz - Technische Hochschule Ingolstadt - 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from net.slayer import SLAYERAudioPipeline
from net.lstm import LSTMAudioPipeline

from codec.pcm import PulseCodeModulation
from codec.pdm import PulseDensityModulation
from codec.bin import BinaryVectorModulation
from codec.grid import GridCellModulation
from visualize.modulation_visualize import ModulationVisualizer


def main():
    # INPUT
    input_signal = PulseCodeModulation().from_file(audiopath, dtype=f'int{bit_depth}')
    #input_signal = PulseCodeModulation().from_array([[int(np.sin(x*0.1)*(2**16/2)) for x in range(5000)]], input_signal.samplerate)

    # ENCODING
    if strategy.lower() == 'binary':
        codec = BinaryVectorModulation(bit_depth, weighted=False)
    elif strategy.lower() == 'wbinary':
        codec = BinaryVectorModulation(bit_depth, weighted=True)
    elif strategy.lower() == 'deltasigma':
        input_signal.oversample(OSR=oversampling)
        codec = PulseDensityModulation(order=1, OSR=oversampling)
    elif strategy.lower() == 'gridcell':
        codec = GridCellModulation(2**bit_depth, seed=seed)
        if new_training:
            codec.new_modules(module_count=60, cells_per_module=10).save_modules(gridcell_path)
        else:
            codec.load_modules_from_file(gridcell_path)
    else: return

    # FEATURE VECTOR (binary / deltasigma / gridcell)
    input_vectors = codec.value2vector(input_signal.get_waveform(wave_format='Channel-Amplitude', mono=mono))
    desir_vectors = input_vectors  # codec.value2vector(desir_signal.get_waveform(wave_format='Amplitude-Channel', mono=mono))

    # NETWORK
    if activate_net:
        # Do the whole ENCODING -> NETWORK -> DECODING process
        network_audio = None
        if network.lower() == 'slayer':
            network_audio = SLAYERAudioPipeline(config_path=config_path, input_size=codec.get_vector_size(),
                                                output_size=codec.get_vector_size(),
                                                samplerate=input_signal.samplerate, mono=mono)
        elif network.lower() == 'lstm':
            network_audio = LSTMAudioPipeline(config_path=config_path, input_size=codec.get_vector_size(),
                                              output_size=codec.get_vector_size(),
                                              samplerate=input_signal.samplerate, mono=mono)

        if new_training:
            network_audio.visualize_training(visualize_training)
            network_audio.train(input_vectors, desir_vectors, n_epochs, checkpoint, training_path)
        else:
            pass
            network_audio.load(training_path)

        # OUTPUT
        output_vectors = network_audio.test(input_vectors, desir_vectors, plot=visualize_result)
    else:
        # To test ENCODING -> DECODING without a network
        output_vectors = input_vectors

    output_waveform = codec.vector2value(output_vectors)
    output_signal = PulseCodeModulation().from_array(output_waveform, input_signal.samplerate, array_format='Channel-Amplitude')        # TODO: PDM has wrong samplerate here!

    if visualize_result:
        if strategy.lower() == 'deltasigma':
            plot = ModulationVisualizer(title='Test Waveform', number_of_plots=3)
            plot.plot_pcm(input_signal, axidx=0, title='Input Signal')
            plot.plot_pdm(codec, axidx=1)
            plot.plot_pdm(codec + 3, axidx=1)
            plot.plot_pcm(output_signal, axidx=2, title='Decoded Signal')
        else:
            plot = ModulationVisualizer(title='Test Waveform', number_of_plots=2)
            plot.plot_pcm(input_signal, axidx=0, title='Input Signal')
            plot.plot_pcm(output_signal, axidx=1, title='Decoded Signal')
        plot.show()

    print('\nPlayback...')
    input_signal.play(volume)
    output_signal.play(volume)


if __name__ == '__main__':
    # ------- CODEC --------
    #strategy = 'BINARY'
    #strategy = 'WBINARY'
    #strategy = 'GRIDCELL'
    strategy = 'DELTASIGMA'

    # ------ NETWORK -------
    network = 'SLAYER'
    #network = 'LSTM'

    seed = 0

    save_name = strategy.lower()
    save_version = '001'
    config_path = f'../config/net_{network.lower()}.yaml'

    audiopath = '../res/audio/input/speech1.wav'
    volume = 50000
    bit_depth = 16
    oversampling = 32       # for Delta Sigma Modulation only
    mono = True

    activate_net = True
    new_training = True
    training_path = f'../save/net/{network.lower()}/{save_name}-{save_version}-{"mono" if mono else "stereo"}.pt'
    new_gridcells = False
    gridcell_path = f'../save/grid/gcmodule-{save_version}-{"mono" if mono else "stereo"}.csv'
    n_epochs = 3000
    checkpoint = 100

    visualize_training = True
    visualize_result = True

    main()
