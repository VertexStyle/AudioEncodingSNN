"""
Converts integer value(s) to a binary list
"""
import numpy as np
import re

from src.codec.modulation import Modulation


class BinaryVectorModulation(Modulation):
    def __init__(self, bit_depth, sign=('0', '1'), weighted=False):
        super().__init__()

        self.bit_depth = bit_depth
        self.scaling = int(2**(bit_depth-1))
        self.sign = sign
        self.weighting = weighted

    def value2vector(self, value, flatten=False, force_depth=None):
        if hasattr(value, "__len__"):
            if flatten:
                bitvec = [self.value2vector(v, force_depth=force_depth).flatten() for v in value]
            else:
                bitvec = [self.value2vector(v, force_depth=force_depth) for v in value]
            return np.array(bitvec, dtype=np.int8)
        else:
            if force_depth:
                bit_depth = force_depth
            else:
                bit_depth = self.bit_depth

            if value < 0:
                vector = self.sign[1] + str(bin(value)).replace('0b', '').replace('-','').zfill(bit_depth-1)
            else:
                vector = self.sign[0] + str(bin(value)).replace('0b', '').zfill(bit_depth-1)

            int_vector = []
            if not self.weighting:
                int_vector = [int(bit) for bit in vector]
            else:
                significance = self.get_significance()
                for bit, repeat in zip(vector, significance):
                    int_vector.extend([int(bit) for _ in range(repeat)])
            return np.array(int_vector, dtype=np.int8)

    def vector2value(self, vector):
        if hasattr(vector, '__len__'):
            vector = list(vector)
            if hasattr(vector[0], '__len__'):
                return [self.vector2value(bit) for bit in vector]

            if not self.weighting:
                bitvector = vector
            else:
                significance = self.get_significance()
                offset = 0
                bitvector = []
                for repeat in significance:
                    bitrepeat = vector[offset:offset+repeat]
                    bit = max(set(bitrepeat), key=bitrepeat.count)
                    bitvector.append(bit)
                    offset += repeat
                    #print(repeat, bitrepeat, bit)

            signbit = 1 if bitvector.pop(0) == int(self.sign[0]) else -1
            return signbit * int(''.join(str(bit) for bit in bitvector), 2)

    def get_vector_size(self):
        return sum(self.get_significance())


    def get_domain(self, dtype=bin):
        if dtype == str:
            domain = []
            for value in range(-self.scaling, self.scaling):
                domain.append(''.join(self.value2vector(value)))
            return domain
        else:
            return [dtype(value) for value in range(-self.scaling, self.scaling)]

    def get_range(self, dtype=None):
        if not dtype:
            return range(-self.scaling, self.scaling)
        elif dtype == str:
            bmin = ''.join([str(bit) for bit in self.value2vector(-self.scaling)])
            bmax = ''.join([str(bit) for bit in self.value2vector(self.scaling - 1)])
            return bmin, bmax
        else:
            return dtype(-self.scaling), dtype(self.scaling-1)

    def get_significance(self):
        if self.weighting:
            return [sg for sg in range(self.bit_depth, 0, -1)]
        else:
            return [1 for sg in range(self.bit_depth)]


if __name__ == '__main__':
    bind = BinaryVectorModulation(16, weighting=True)
    print("Domain:", bind.get_range(dtype=int), bind.get_range(dtype=str))

    val = [[1,2**15], [600,3]]
    print("In-Value:", val)

    vec = bind.value2vector(val, flatten=False)
    print("Vector:", vec)

    dec = bind.vector2value(vec)
    print("Out-Value:", dec)

    sig = bind.get_significance()
    print("Significance:", sig)
    
    