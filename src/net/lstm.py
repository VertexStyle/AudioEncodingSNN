"""From https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html"""

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device as Device
import time

from src.net.audiopipe import AudioPipeline


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, recurr_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.audio_embedding = nn.Embedding(recurr_dim, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

    def forward(self, spikes):
        embeds = self.audio_embedding(spikes)
        lstm_out, _ = self.lstm(embeds.view(len(spikes), 1, -1))
        prob_out = self.hidden2out(lstm_out.view(len(spikes), -1))
        #scores = F.log_softmax(spike_out, dim=1)
        return prob_out


class LSTMAudioPipeline(AudioPipeline):
    device: Device
    net: LSTM
    error: nn.NLLLoss

    def __init__(self, config_path=None, input_size=None, output_size=None, samplerate=None, mono=False):
        super().__init__()

        # CUDA Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            raise RuntimeError("No CUDA device found.")

        if input_size and output_size and samplerate and config_path:
            # LSTM Parameters
            with open(config_path, 'r') as param_file:
                self.params = yaml.safe_load(param_file)

            # Initialise Network
            channels = 1 if mono else 2
            self.inp_dim = input_size * channels  # Format: (Width, Height, Channel)
            self.out_dim = output_size * channels

            self.net = LSTM(self.inp_dim, self.params['layer']['hiddenDim'],
                            self.params['layer']['recurrDim'], self.out_dim).to(self.device)
            self.error = nn.NLLLoss()
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params['training']['learningRate'],
                                              amsgrad=self.params['training']['amsgrad'])
            # TODO: optim.SGD(model.parameters(), lr=0.1)

            self.sample_time = self.params['simulation']['Ts']
            self.sample_time_length = int(self.params['simulation']['tSample'] / self.params['simulation']['Ts'])

            self.training_plot = None


    def train(self, input_vectors, desire_vectors, n_epochs=10000, checkpoint=500, save_path=None, input_format='Channel-Time-Neuron'):
        order = self.get_input_order(input_format)
        input_vectors = np.moveaxis(np.array(input_vectors), (0, 1, 2), order)
        desire_vectors = np.moveaxis(np.array(desire_vectors), (0, 1, 2), order)

        # Predictions before training
        with torch.no_grad():
            input_tensor = self.get_tensor(input_vectors[0][0])
            prediction = self.net(input_tensor)
            print(prediction)

        start_time = time.time()
        run_time = start_time
        loss_log = []
        best_loss = None
        for epoch in range(n_epochs):
            loss = None
            for input_vector, desire_vector in zip(input_vectors[0], desire_vectors[0]):    # TODO: currently only mono
                self.net.zero_grad()

                # Get tensors
                input_tensor = self.get_tensor(input_vector)
                desir_tensor = self.get_tensor(desire_vector, dtype=torch.long)

                # Forward pass
                tag_scores = self.net(input_tensor)

                # Loss
                loss = self.error(tag_scores, desir_tensor)
                loss_log.append(loss)

                # Backard pass
                loss.backward()
                self.optimizer.step()

            # Log & Save
            if epoch % checkpoint == 0:
                print(f"Epoch: {epoch}/{n_epochs} Loss: {loss:.4f}")

                elapsed = time.time()
                print(f"Time elapsed: {round(elapsed - run_time, 2)} seconds")
                run_time = elapsed

                self.update_training_plot(loss_log)

            if not best_loss:
                best_loss = loss

            if save_path and loss < best_loss:
                torch.save(self.net.state_dict(), save_path)


        # Predictions after training
        with torch.no_grad():
            inputs = self.get_tensor(input_vectors[0][0])
            prediction = self.net(inputs)
            print(prediction)

    def test(self, input_vectors, desire_vectors, input_format='Channel-Time-Neuron'):
        order = self.get_input_order(input_format)
        input_vectors = np.moveaxis(np.array(input_vectors), (0, 1, 2), order)
        desire_vectors = np.moveaxis(np.array(desire_vectors), (0, 1, 2), order)



    def get_tensor(self, inputs, dtype=torch.int):
        tensor = torch.tensor(inputs, dtype=dtype, device=self.device)
        return tensor
