import numpy as np
import matplotlib.pyplot as plt

from src.codec.pcm import PulseCodeModulation
from src.codec.bin import BinaryVectorModulation
from src.codec.grid import GridCellModulation
from src.codec.pdm import PulseDensityModulation



def benchmark_codecs():
    input_audio = PulseCodeModulation().from_file(audiopath, dtype=f'int{bit_depth}', start=start, end=end)
    wave = input_audio.get_waveform(wave_format='Channel-Amplitude', mono=True)
    desire = wave.flatten()

    bin = BinaryVectorModulation(bit_depth)
    wbn = BinaryVectorModulation(bit_depth, weighted=True)
    gcm = GridCellModulation(2**bit_depth, seed=0).new_modules(module_count=80, cells_per_module=10)
    pdm = PulseDensityModulation(order=6, OSR=oversampling)

    errors = {}
    vsizes = {}
    audios = {}

    print('{:^30}  {:>5} || {:>21} | {:>21} | {:>21} | {:>21} | {:>21} | {:>21}'.format('-- ERRORS (min, mean, max) --', 'SIZE', 'p=0.0', 'p=0.1', 'p=0.2', 'p=0.3', 'p=0.4', 'p=0.5'))
    for name, codec in {"Two's Complement": bin, "Weighted Two's Complement": wbn, "Grid Cell Modules": gcm, "Delta Sigma Modulation": pdm}.items():
        if codec == pdm:
            input_audio.oversample(OSR=oversampling)
            wave_os = input_audio.get_waveform(wave_format='Channel-Amplitude', mono=True)
            encoding = np.array(codec.value2vector(wave_os))
        else:
            encoding = np.array(codec.value2vector(wave))

        error_list = []
        audio_list = []

        for p in reversed(noise_probs):
            noise = np.random.binomial(1, p, encoding.shape)                                        # generate random noise with binomial distribution
            encoding_noisy = np.array(np.logical_xor(encoding, noise), dtype=np.int8)               # add noise to feature vector

            decoding = np.array(codec.vector2value(encoding_noisy))
            output_audio = PulseCodeModulation().from_array(decoding, input_audio.samplerate, array_format='Channel-Amplitude')

            decode = output_audio.get_waveform().flatten()

            plt.plot(decode, label=f'p={p}', alpha=1-p*1.7, linewidth=0.7)

            difference = np.abs(desire - decode)
            difference_float = (difference / 2**(bit_depth-1))

            error_list = [(round(np.min(difference_float),4), round(np.mean(difference_float),4), round(np.max(difference_float),4))] + error_list
            audio_list = [output_audio] + audio_list


        plt.plot(desire, label='Original', c='black')
        plt.legend()
        plt.show()

        print('{:>30}: {:>5} || {:>21} | {:>21} | {:>21} | {:>21} | {:>21} | {:>21}'.format(name, codec.get_vector_size(), *(str(e[1]) for e in error_list)))

        errors[name] = error_list
        vsizes[name] = codec.get_vector_size()
        audios[name] = audio_list


if __name__ == '__main__':
    audiopath = '../../res/audio/input/test1.wav'
    start = 5000
    end = 5500

    noise_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    bit_depth = 16
    oversampling = 32

    benchmark_codecs()
