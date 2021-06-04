import re
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import torch

class AudioPipeline(object):
    def __init__(self):
        self.sample_time_length = None
        self.sample_time = None
        self.training_plot = None
        self.net = None

        self.logpath = None
        self.title = ''
        self.subtitle = ''

    @abstractmethod
    def train(self, input_vectors, desire_vectors, n_epochs=10000, checkpoint=500, save_path=None, input_format='Channel-Time-Neuron'):
        pass

    @abstractmethod
    def test(self, input_vectors, desire_vectors, input_format='Channel-Time-Neuron'):
        pass

    @abstractmethod
    def get_tensor(self, inputs, batches=None):
        return None, None


    def load(self, file_path):
        # Load network parameters
        self.net.load_state_dict(torch.load(file_path))

    def visualize_training(self, visualize=True, logpath=None, title='Training Loss', subtitle=''):
        self.title = title
        self.subtitle = subtitle
        if logpath:
            self.logpath = logpath
        if visualize:
            plt.ion()

            fig, ax = plt.subplots()
            plt.title('Training Loss')
            plt.xlabel('Time')
            plt.ylabel('Loss')
            #ax.set_xlabel('Time')
            #ax.set_ylabel('Loss')

            self.training_plot = (fig, ax)
        else:
            self.training_plot = None

    def visualize_results(self, input_frame, desired_frame, predict_frame, Nin, Nout, stats=None):
        plt.figure(1)

        inpAER = np.argwhere(input_frame.reshape((Nin, self.sample_time_length)).cpu().data.numpy() > 0)
        prdAER = np.argwhere(predict_frame.reshape((Nout, self.sample_time_length)).cpu().data.numpy() > 0)
        if desired_frame is not None:
            desAER = np.argwhere(desired_frame.reshape((Nout, self.sample_time_length)).cpu().data.numpy() > 0)

            plt.plot(desAER[:, 1], desAER[:, 0], 'o', label='desired')
        plt.plot(prdAER[:, 1], prdAER[:, 0], '.', label='actual')

        plt.title('Training Loss')
        plt.xlabel('time')
        plt.ylabel('neuron ID')
        plt.legend(loc='upper right')

        if stats:
            plt.figure(2)
            plt.semilogy(stats.training.lossLog)
            plt.title('Training Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')

        plt.show()

    def load_features(self, inputs, desired=None, reorder=(0, 1, 2), batches=None):
        inp_reorder = np.moveaxis(np.array(inputs), (0, 1, 2), reorder)
        
        inp_tensor, inp_event = self.get_tensor(inp_reorder, batches=batches)
        #showTD(event)

        if desired is not None:
            des_reorder = np.moveaxis(np.array(desired), (0, 1, 2), reorder)
            des_tensor, des_event = self.get_tensor(des_reorder, batches=batches)

            return inp_tensor, des_tensor
        return inp_tensor, None

    def update_training_plot(self, loss_log, acc_log=None):
        if self.training_plot:
            fig, ax = self.training_plot
            plt.plot(loss_log, label='Loss')
            if acc_log:
                plt.plot(acc_log, label='Accuracy')
            plt.legend()

            plt.title(self.subtitle)
            plt.suptitle(self.title, fontsize=15, y=1)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.draw()
            if self.logpath:
                plt.savefig(self.logpath)
                with open(self.logpath + '.txt', 'w') as f:
                    f.write('\n'.join((str(v) for v in loss_log)))
            plt.pause(0.0001)
            plt.clf()


    @staticmethod
    def get_input_order(input_format):
        return tuple(int(pos) for pos in re.split(r'\D+', input_format.upper().replace('CHANNEL', '0').replace('NEURON','1').replace('TIME', '2')))

