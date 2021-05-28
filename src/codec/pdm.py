from deltasigma import sinc_decimate, synthesizeNTF, simulateDSM
from src.codec.modulation import Modulation
import numpy as np

class PulseDensityModulation(Modulation):
    def __init__(self, order=6, OSR=64, optimize=1, sinc_order=16):
        self.bitstream = None
        self.OSR = OSR
        self.order = order
        self.optimize = optimize
        self.sinc_order = sinc_order

        # self.ABCD = np.array([[1., 0., 1., -1.], [1., 1., 1., -2.], [0., 1., 0., 0.]])

    def value2vector(self, oversampled_wave, flatten=False):
        """Convert a PCM signal array to a Delta Sigma modulated bitstream array."""
        NTF = synthesizeNTF(order=self.order, osr=self.OSR, opt=self.optimize)
        bitstream, xn, xmax, y = simulateDSM(oversampled_wave, NTF)
        bitstream = np.array([[bitstream]], dtype=np.int8)
        bitstream = np.expand_dims(bitstream, axis=2)         # not sure why axis 2 but works
        self.bitstream = bitstream.flatten()
        return bitstream

    def vector2value(self, bitstream):
        """Convert a Delta Sigma modulated bitstream back to a PCM signal."""
        bitstream = np.array(bitstream)
        if hasattr(bitstream, '__len__'):
            if hasattr(bitstream[0], '__len__'):
                return [self.vector2value(np.array(bit).flatten()) for bit in bitstream]
            else:
                signal = sinc_decimate(bitstream, self.sinc_order, self.OSR)
                return signal

    def get_vector_size(self):
        return 1

    def from_pcm(self, reference):
        """Old version of value2vector defined above. Might still be useful."""
        self.OSR = reference.OSR
        NTF = synthesizeNTF(order=self.order, osr=self.OSR, opt=self.optimize)
        self.bitstream, xn, xmax, y = simulateDSM(reference.waveform, NTF)  # PCM zu (Delta-Sigma Moduliertem) Bitstream
        return self
