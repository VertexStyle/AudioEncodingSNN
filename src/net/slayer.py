import re
import time
import torch
from torch.nn import Module
from torch import device as Device
import numpy as np
import matplotlib.pyplot as plt
import random as r

import slayerSNN as snn
from slayerSNN import loihi as SpikeLayer
from slayerSNN.learningStats import learningStats as LearningStats
from slayerSNN.slayerParams import yamlParams as YamlParams
from slayerSNN import quantize as quantize_params
from slayerSNN import optimizer as optimize
from slayerSNN.spikeFileIO import spikeArrayToEvent, showTD, animTD

from src.net.audiopipe import AudioPipeline


class SLAYER(torch.nn.Module):
    def __init__(self, inp_dim, out_dim, params):
        super(SLAYER, self).__init__()

        # Initialize SLAYER
        self.slayer = SpikeLayer(params['neuron'], params['simulation'])

        # Define Layers
        input_size = inp_dim
        layer_sizes = [int(params['hidden'][lyr]['dim']) for lyr in range(len(params['hidden']))] + [out_dim]

        self.layers = torch.nn.ModuleList()
        for neuron_count in layer_sizes:
            self.layers.append(self.slayer.dense(input_size, neuron_count))
            input_size = neuron_count

    def forward(self, spikes):
        for fc in self.layers:
            spikes = self.slayer.spikeLoihi(fc(spikes))
            spikes = self.slayer.delayShift(spikes, 1)
        return spikes


class SLAYERAudioPipeline(AudioPipeline):
    params: YamlParams
    device: Device
    net: SLAYER
    error: Module

    def __init__(self, config_path=None, input_size=None, output_size=None, samplerate=None, mono=False, layer_sizes=None):
        super().__init__()

        # CUDA Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            raise RuntimeError("No CUDA device found.")

        if input_size and output_size and samplerate and config_path:
            # Slayer Parameters
            self.params = snn.params(config_path)
            if layer_sizes:
                for lyr_i, lyr in enumerate(layer_sizes):
                    self.params['hidden'][lyr_i]['dim'] = lyr

            # Initialise Network
            channels = 1 if mono else 2
            self.inp_dim = input_size * channels  # Format: (Width, Height, Channel)
            self.out_dim = output_size * channels
            self.net = SLAYER(self.inp_dim, self.out_dim, self.params).to(self.device)
            self.error = snn.loss(self.params, SpikeLayer).to(self.device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params['training']['learningRate'],
                                              amsgrad=self.params['training']['amsgrad'])

            self.sample_time = self.params['simulation']['Ts']
            if self.params['simulation']['tSample'] > 0:
                self.sample_time_length = int(self.params['simulation']['tSample'] / self.params['simulation']['Ts'])
            else:
                self.sample_time_length = None

            self.training_plot = None

    def train(self, input_vectors, desire_vectors, n_epochs=10000, checkpoint=500, save_path=None, input_format='Channel-Time-Neuron'):
        # Prepare Feature Vectors           TODO: tensor definition inside training???
        order = self.get_input_order(input_format)
        time_samples = input_vectors.shape[order.index(2)]
        if not self.sample_time_length:
            self.params['simulation']['tSample'] = time_samples
            self.sample_time_length = int(time_samples / self.params['simulation']['Ts'])

        batch_size = self.params['training']['timeBatchSize']
        if batch_size == -1:
            batch_size = None
        input_tensor, desire_tensor = self.load_features(input_vectors, desire_vectors, reorder=order,
                                                         batches=batch_size)
        print('>>> Tensor Shape:', input_tensor.shape, end='')
        # Run Training
        stats = LearningStats()
        start_time = time.time()
        run_time = start_time

        for epoch in range(n_epochs):
            # Forward
            predicted = self.net.forward(input_tensor)
            errors = self.error.spikeTime(predicted, desire_tensor)

            # Stats
            stats.training.numSamples = 1
            stats.training.lossSum = errors.cpu().data.item()
            stats.training.update()

            # Log & Save
            if epoch % checkpoint == 0:
                stats.print(epoch)

                elapsed = time.time()
                print(f"Time elapsed: {round(elapsed - run_time, 2)} seconds")
                run_time = elapsed

                self.update_training_plot(stats.training.lossLog) #, stats.training.accuracyLog)

            if save_path and stats.training.bestLoss:
                torch.save(self.net.state_dict(), save_path)

            if errors < 1e-5: break

            # Backward
            self.optimizer.zero_grad()
            errors.backward()
            self.optimizer.step()

        print(f"\n>>> Training took {round((time.time() - start_time)/60, 2)} minutes")

    def test(self, input_vectors, desired_vectors=None, input_format='Channel-Time-Neuron', plot=False, logpath=None, use_batches=False):
        # Prepare Feature Vectors
        order = self.get_input_order(input_format)
        time_samples = input_vectors.shape[order.index(2)]
        #print(order, input_vectors.shape)
        if not self.sample_time_length:
            self.params['simulation']['tSample'] = time_samples
            self.sample_time_length = int(time_samples / self.params['simulation']['Ts'])
        
        if use_batches:
            batch_size = self.params['training']['timeBatchSize']
            if batch_size == -1:
               batch_size = None
        else:
            batch_size = None
        input_tensor, desire_tensor = self.load_features(input_vectors, desired_vectors, reorder=order, batches=batch_size)
        print('>>> Tensor Shape:', input_tensor.shape, end='')

        # Run test
        prediction_tensor = self.net.forward(input_tensor)
        if plot:
            self.visualize_results(input_tensor, desire_tensor, prediction_tensor, self.inp_dim, self.out_dim)

        prediction_raw = prediction_tensor.cpu().data.numpy()
        prediction = np.array(np.concatenate(prediction_raw, axis=1), dtype=np.int8)
        prediction = np.concatenate(prediction, axis=1)

        return np.moveaxis(prediction, order, (0, 1, 2))

    def get_tensor(self, inputs, batches=None):
        """
        Slayer Documentation: https://bamsumit.github.io/slayerPytorch/build/html/slayer.html
        Format of input vectors is   (Channel, Amplitude, Time)
        Format of tensor needs to be (Batch, Channel, Height, Width, Time)
        Number of neurons = Channel * Height * Width
        """
        if not batches:
            batches = self.sample_time_length

        inputs = np.concatenate(inputs, axis=0)     # merge stereo channels two one channel (without loosing stereo information)
        inputs = np.expand_dims(inputs, axis=0)     # add back channel dimension as empty
        event = spikeArrayToEvent(inputs)
        tensor = event.toSpikeTensor(torch.zeros((1, 1, inputs.shape[1], self.sample_time_length)), samplingTime=self.sample_time)

        # Reshape tensor to ignore spatial dimensions (Height, Width). Time will be split into batches. All neurons will be in channel dimension.
        tensor = tensor.reshape((1, inputs.shape[1], 1, -1, batches))   # split time into batches
        tensor = tensor.swapaxes(0, 3)                                  # swap axes so batches are in the right position
        spike_tensor = tensor.to(self.device)                           # transfer to GPU
        return spike_tensor, event


if __name__ == '__main__':
    # Tensor generation test

    ar = np.array([[[1,1.1,1.2],[2,2.1,2.2],[3,3.1,3.2]], [[4,4.1,4.2],[5,5.1,5.2],[6,6.1,6.2]]])   # (Channel, Amplitude, Time)
    print("RAW\n", ar, ar.shape)
    print()

    ar = np.concatenate(ar, axis=0)         # merge channels
    ar = np.expand_dims(ar, axis=0)         # add back channel dimension as empty
    print("MERGE\n", ar, ar.shape)
    print()

    zr = np.random.randint(0,9,(1, 1, ar.shape[1], 2500))       # create base shape
    print("ZEROS\n", zr, zr.shape)
    print()

    tr = zr.reshape((1, ar.shape[1], 1, -1, 500))               # split time into batches
    tr = np.swapaxes(tr, 0, 3)                                  # swap axes so batches are at the right position
    print("RESHAPE\n", str(tr).replace('\n\n\n', '\n'), tr.shape)
    print()
