"""
Create binary vectors out of integer vectors using Grid-Cells
"""
import time
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import operator as op
import csv

from src.codec.modulation import Modulation
from src.codec.pcm import PulseCodeModulation


class GridCellModulation(Modulation):
    def __init__(self, vector_size=None, interval=None, seed=None):
        """
        Either scaling or interval needs to be given.
        """
        super().__init__()

        if interval:
            self.scaling = int(abs(interval[1] - interval[0]))
            self.interval = interval
        elif vector_size:
            self.scaling = int(vector_size)
            self.interval = (-self.scaling//2, self.scaling//2 + 1)     # center interval around zero

        self.module_count = 10
        self.cells_per_module = 10
        np.random.seed(seed)

        self.grid_cell_modules = []
        self.module_config = []

        self.receptive_matrix = np.array([])

        self.plot = True

        self.last_decoded_value = None
        self.last_decoded_slope = None
        self.mean_slope = 0
        self.repairs = 0

    def value2vector(self, value, flatten=False):
        """
        Converts an integer value to a grid cell vector representation.
        """
        if hasattr(value, "__len__"):
            if flatten:
                gcvec = [self.value2vector(v).flatten() for v in value]
            else:
                gcvec = [self.value2vector(v) for v in value]
            return np.array(gcvec, dtype=np.int8)
        else:
            gcvec = np.zeros(self.get_cell_count(), dtype=np.int8)
            for module_index, gcm in enumerate(self.grid_cell_modules):
                active_index = gcm.get_activation_index(int(value)) + module_index * self.cells_per_module
                try:
                    gcvec[active_index] = 1
                except IndexError:
                    continue
            return np.array(gcvec, dtype=np.int8)

    def vector2value(self, vector, interval=None, find_all=False, lookup=True, plot=False, max_slope_change=1000):
        """
        Converts a grid cell vector representation back into an integer value
        """
        if self.plot and not plot:
            self.plot = False

        if hasattr(vector[0], '__len__'):
            return [self.vector2value(bit, plot=plot) for bit in vector]

        if not interval:
            interval = self.interval
        itv_size = int(abs(interval[1] - interval[0]))
        field_values = np.arange(*interval)

        # build matrix out of all receptive fields
        cell_count = self.get_cell_count()
        active_cells = np.argwhere(np.array(vector) == 1).flatten()
        active_cells = active_cells[active_cells < cell_count]

        if lookup:
            # build matrix as lookup table with all cells (faster)
            if self.receptive_matrix.shape[0] > 0:
                receptive_mtrx = self.receptive_matrix
            else:
                receptive_mtrx = np.zeros((cell_count, itv_size), dtype=np.int8)
                for i in range(cell_count):
                    module_index = i // self.cells_per_module
                    cell_index = i % self.cells_per_module
                    receptive_mtrx[i] = self.grid_cell_modules[module_index].get_receptive_field(cell_index, field_values)
                self.receptive_matrix = receptive_mtrx

            active_values = receptive_mtrx[active_cells].sum(axis=0)

        else:
            # build matrix only with active cells (slow)
            receptive_mtrx = np.zeros((len(active_cells), itv_size), dtype=np.int8)
            for i, active_index in enumerate(active_cells):
                module_index = active_index // self.cells_per_module
                cell_index = active_index % self.cells_per_module
                receptive_mtrx[i] = self.grid_cell_modules[module_index].get_receptive_field(cell_index, field_values)

            # find most active column in matrix and map the index to the interval
            active_values = receptive_mtrx.sum(axis=0)

        if self.plot:
            sort_val = list(reversed(np.sort(active_values)))
            plt.plot(sort_val)
            plt.fill([0] + sort_val + [0], alpha=0.5)
            plt.show()
            self.plot = False

        if find_all:
            candidates = np.argwhere(active_values == np.max(active_values))
            decoded = np.array(op.itemgetter(*candidates)(field_values)).flatten()
        else:
            candidate = np.argmax(active_values)
            decoded = field_values[candidate]

            # if the slope changes too drastically pick the last one to prevent spikes
            if max_slope_change:
                if self.last_decoded_value is not None and self.last_decoded_slope is not None:
                    current_slope = decoded - self.last_decoded_value
                    slope_difference = abs(current_slope - self.last_decoded_slope)
                    mean_slope = self.mean_slope / (self.repairs + 1)

                    theta = 1 / (self.repairs+1)
                    if slope_difference > mean_slope * 2:
                        decoded = int(theta * (self.last_decoded_value + np.ceil(self.last_decoded_slope)) + (1-theta) * decoded)
                        self.repairs += 1
                    else:
                        # found a correct value
                        self.last_decoded_slope = current_slope
                        self.repairs = 0
                        self.mean_slope = 0

                    offset = 0.2
                    self.mean_slope += (theta-offset) * mean_slope + (1+offset-theta) * slope_difference
                    self.last_decoded_value = decoded

                elif self.last_decoded_value is not None:
                    self.last_decoded_slope = decoded - self.last_decoded_value
                    self.last_decoded_value = decoded
                else:
                    self.last_decoded_slope = 0
                    self.last_decoded_value = decoded

        return decoded


    def get_vector_size(self):
        return self.get_cell_count()

    def new_modules_NEW(self, module_count=5, cells_per_module=20):
        self.module_count = module_count
        self.cells_per_module = cells_per_module

        for _ in range(module_count):
            radius = min(np.abs(np.random.normal(0,self.scaling/64) + self.scaling/64), self.scaling)
            #print(radius)
            offset = np.random.randint(0, cells_per_module * radius)
            self.grid_cell_modules.append(GridCellModule(cell_count=cells_per_module, cell_radius=radius, offset=offset))
            # So modules can be saved later
            self.module_config.append((module_count, cells_per_module, radius, offset))
        return self

    def new_modules(self, module_count=5, cells_per_module=20):
        self.module_count = module_count
        self.cells_per_module = cells_per_module

        radius_picker = self.scaling // 2
        divisor = 0.3
        for _ in range(module_count):
            radius_picker = radius_picker // min(2, 1 + divisor)
            if radius_picker <= 16:
                radius_picker = self.scaling // 2
                divisor = 0.3
            else:
                divisor *= 1.1
            radius = np.random.randint(1, radius_picker)
            offset = np.random.randint(0, cells_per_module * radius)
            self.grid_cell_modules.append(GridCellModule(cell_count=cells_per_module, cell_radius=radius, offset=offset))
            # So modules can be saved later
            self.module_config.append((module_count, cells_per_module, radius, offset))
        return self

    def save_modules(self, path):
        with open(path, 'w', newline='') as cf:
            module_writer = csv.writer(cf, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for mdl_param in self.module_config:
                module_writer.writerow(mdl_param)
        return self

    def load_modules_from_file(self, path):
        with open(path, newline='') as cf:
            module_reader = csv.reader(cf, delimiter=';', quotechar='|')
            self.module_config = []
            for mdl_param in module_reader:
                self.module_config.append(tuple(int(v) for v in mdl_param))
        self.load_modules_from_config()
        return self

    def load_modules_from_config(self, config=None):
        if config:
            self.module_config = config

        self.module_count = self.module_config[0][0]
        self.cells_per_module = self.module_config[0][1]
        for mdl_data in self.module_config:
            radius = mdl_data[2]
            offset = mdl_data[3]
            self.grid_cell_modules.append(GridCellModule(cell_count=self.cells_per_module, cell_radius=radius, offset=offset))
        return self

    def get_cell_count(self):
        return self.cells_per_module * self.module_count

    def visualize_grid_cells(self, interval=None):
        if not interval:
            interval = self.interval

        patterns = []
        for gcm in self.grid_cell_modules:
            for p in gcm.visualize_module(interval, plot=False):
                patterns.append(p.toarray()[0])

        if len(patterns) > 0:
            plt.matshow(patterns, cmap='binary')

            scale = (interval[1] - interval[0]) // 10
            ticks = np.arange(0, (interval[1] - interval[0]), scale)
            labels = np.arange(*interval, scale)

            plt.xlabel('Encoded Value')
            plt.ylabel('Activated Grid Cells')

            plt.xticks(ticks, labels)
            plt.show()


class GridCellModule(object):
    """
    As proposed by Numenta.
    """
    def __init__(self, cell_count=32, cell_radius=16, offset=0):
        self.cell_count = cell_count
        self.cell_radius = cell_radius
        self.offset = offset

    def get_receptive_field(self, cell_index, field_values):
        """
        Returns a binary vector that indicates which of the cell fires for what value.
        """
        field = np.zeros_like(field_values)
        field[self.get_activation_index(field_values) == cell_index] = 1
        return field

    def get_activation_index(self, val):
        """
        Returns index of the cell that fired for a given value.
        Calculating cell activation with modulo avoids lookup table.
        Eigenvector based codec.
        """
        return ((val + self.offset) // self.cell_radius) % self.cell_count

    def visualize_module(self, interval, plot=True):
        field_values = np.arange(*interval)
        cells = sp.lil_matrix(np.zeros((self.cell_count, len(field_values))))
        for ccnt in range(self.cell_count):
            act = self.get_receptive_field(ccnt, field_values)
            cells[ccnt] = act

        if plot:
            plt.matshow(cells.toarray(), cmap='binary')
            plt.show()

        return cells


if __name__ == '__main__':
    nois = []
    cell = []
    errs = []
    tmes = []

    def optimize_plot():
        gc_pipe = GridCellModulation(vector_size=2 ** 16, seed=0).new_modules(module_count=50, cells_per_module=50)
        gc_pipe.visualize_grid_cells(interval=(-100, 101))

        nvecs = []
        pnoises = []
        for n in range(0,20,2):
            pnoise = n / 40
            nvecs.append(np.random.choice([0,1], gc_pipe.get_cell_count(), p=[1-pnoise, pnoise]))
            pnoises.append(pnoise)


        for noise, pnoise in zip(nvecs, pnoises):
            for c in range(20, 10, -4):
                gc_pipe = GridCellModulation(vector_size=2 ** 16, seed=0).new_modules(c, c)

                err = 0
                ets = 0
                cnt = 0

                for test_val in range(-10462, 10000, 135):
                    st = time.time()

                    gc_vector = gc_pipe.value2vector(test_val)
                    out_value = gc_pipe.vector2value(gc_vector + noise[:gc_pipe.get_cell_count()], lookup=True, plot=False, find_all=False)

                    ets += time.time() - st
                    cnt += 1

                    err = abs(test_val-out_value)

                tms = round(ets / cnt, 6)

                # Stats and sample    -> TODO: try noise also to make 1s to 0s
                print(f"Time: {tms} sec. - Error: {err/cnt} - (Cells: {c*c}, Noise: {pnoise})")
                print("In-Value:", test_val)
                print("ORIGI-Vector:", gc_vector.tolist(), len(gc_vector))
                print("NOISE-Vector:", list(gc_vector + noise[:gc_pipe.get_cell_count()]))
                print("Out-Value:", out_value)
                print()


                nois.append(pnoise)
                cell.append(c*c)
                errs.append(err/cnt)
                tmes.append(tms)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.scatter(slst, nlst, heig)

        surf = ax.plot_trisurf(cell, nois, errs, linewidth=1, color='white')
        scat = ax.scatter(cell, nois, errs, c=np.array(tmes) * np.max(tmes), s=35)
        fig.colorbar(scat)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        fig.tight_layout()

        ax.set_xlabel('Cell Count')
        ax.set_ylabel('Noise')
        ax.set_zlabel('Error')
        plt.show()

    def example():
        pnoise = 0.3
        bit_depth = 16
        size = 2**bit_depth
        audiopath = '../../res/audio/input/test1.wav'

        gcm = GridCellModulation(vector_size=size, seed=0).new_modules(module_count=60, cells_per_module=10)

        #space = 25
        #gcm = GridCellModulation(interval=(-space,space+1), seed=None).load_modules_from_config([(4,6,7,-9),(4,6,2,4),(4,6,13,14),(4,6,1,2)])

        gcm.visualize_grid_cells()

        #val = [[int(np.sin(x*0.1)*(size/2)) for x in range(500)]]
        val = PulseCodeModulation().from_file(audiopath, dtype=f'int{bit_depth}').get_waveform(wave_format='Channel-Amplitude', mono=True)[:,5000:5500]
        print("In-Value:", val)


        vec = gcm.value2vector(val, flatten=False)
        #print("Vector:", list(vec[0]))

        noise = np.random.choice([0, 1], vec.shape, p=[1 - pnoise, pnoise])
        noise_vec = np.logical_xor(vec, noise)

        dec = gcm.vector2value(noise_vec, plot=True)
        print("Out-Value:", dec)

        plt.matshow(vec[0].T, cmap='binary')
        plt.xlabel('Sample')
        plt.ylabel('Grid Cell Encoding')
        plt.show()

        plt.plot(val[0], label='Truth')
        plt.plot(dec[0], label='Decode')
        plt.title(f'Noise: {pnoise}, Modules: {gcm.module_count}, Cells per Module: {gcm.cells_per_module}')
        plt.legend()
        plt.show()

    def distribution_normal_test():
        from collections import defaultdict
        import matplotlib.pyplot as plt
        values = defaultdict(float)
        scaling = 2 ** 16
        for i in range(100000):
            value = int((min(np.abs(np.random.normal(scaling / 16, scaling / 16)), scaling)))
            values[value] += 1
        plt.bar(values.keys(), values.values())
        plt.show()
    
    #distribution_normal_test()
    #example()
    #optimize_plot()
