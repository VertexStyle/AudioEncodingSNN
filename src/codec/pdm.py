from deltasigma import sinc_decimate, synthesizeNTF, simulateDSM
from src.codec.modulation import Modulation
import numpy as np

class PulseDensityModulation(Modulation):
    def __init__(self, order=6, OSR=64, optimize=1, sinc_order=16):
        self.OSR = OSR
        self.order = order
        self.optimize = optimize
        self.sinc_order = sinc_order

    def value2vector(self, oversampled_wave, flatten=False):
        """Convert a PCM signal array to a Delta Sigma modulated bitstream array."""
        NTF = synthesizeNTF(order=self.order, osr=self.OSR, opt=self.optimize)
        bitstream, xn, xmax, y = simulateDSM(oversampled_wave, NTF)
        bitstream[bitstream < 0] = 0                            # simulation outputs in interval [-1,1] -> convert to [0,1]
        bitstream = np.array([[bitstream]], dtype=np.int8)
        bitstream = bitstream.swapaxes(1,2)
        # bitstream = np.expand_dims(bitstream, axis=2)
        return bitstream

    def vector2value(self, bitstream):
        """Convert a Delta Sigma modulated bitstream back to a PCM signal."""
        bitstream = np.array(bitstream)
        if hasattr(bitstream, '__len__'):
            if hasattr(bitstream[0], '__len__'):
                bitstream[bitstream == 0] = -1                  # convert from interval [0,1] to [-1,1]
                bitstream = bitstream.reshape(1, -1, 1)

                output_signal = np.array([self.vector2value(np.array(bit).flatten()) for bit in bitstream])
                print(output_signal.shape)
                return output_signal
            else:
                signal = sinc_decimate(bitstream, self.sinc_order, self.OSR)
                return signal

    def get_vector_size(self):
        return 1
