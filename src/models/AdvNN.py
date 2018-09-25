import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torch.nn.utils
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np


class AdvNN(nn.Module):
    def __init__(self, task_in_size, task_hid_size, task_out_size, adv_in_size, adv_hid_size, \
            adv_out_size, adv_count, vocab_size, dropout, lstm_size, adv_depth=1, \
            rnn_dropout=0.0, rnn_type='lstm'):
        super(AdvNN, self).__init__()

        if rnn_type == 'lstm':
            self._rnn = nn.LSTM(input_size=task_in_size, hidden_size=task_hid_size, num_layers=lstm_size, \
                    bias=True, batch_first=True, dropout=rnn_dropout, bidirectional=False)
        elif rnn_type == 'gru':
            self._rnn = nn.GRU(input_size=task_in_size, hidden_size=task_hid_size, num_layers=lstm_size, \
                    bias=True, batch_first=True, dropout=rnn_dropout, bidirectional=False)
        else:
            self._rnn = nn.RNN(input_size=task_in_size, hidden_size=task_hid_size, num_layers=lstm_size, \
                    nonlinearity='tanh', bias=True, batch_first=True, dropout=rnn_dropout, bidirectional=False)

        self.encoder = nn.Embedding(vocab_size + 1, task_in_size)

        task = []
        task.append(nn.Linear(task_in_size, task_hid_size, bias=True))
        task.append(nn.Dropout(p=dropout))
        task.append(nn.Tanh())
        task.append(nn.Linear(task_hid_size, task_out_size, bias=True))
        self.task = nn.Sequential(*task)

        self.advs = []
        for i in range(adv_count):
            adv = []
            for j in range(adv_depth):
                adv.append(nn.Linear(adv_in_size, adv_hid_size, bias=True))
                adv.append(nn.Dropout(p=dropout))
                adv.append(nn.Tanh())
            adv.append(nn.Linear(adv_hid_size, adv_out_size, bias=True))
            self.advs.append(nn.Sequential(*adv).cuda())


        self._hid_dim = task_hid_size
        self._in_dim = task_in_size
        self._adv_count = adv_count
        self._adv_depth = adv_depth
        self._dropout = nn.Dropout(p=dropout)
        self._rnn_dropout =nn.Dropout(p=rnn_dropout)
        self._rnn_type = rnn_type
        self._rnn_layers = lstm_size


    def init_hidden(self, batch_size=32):
        weight = next(self.parameters())
        if self._rnn_type == 'lstm':
            return (weight.new_zeros(self._rnn_layers, batch_size, self._hid_dim).cuda(), \
                    weight.new_zeros(self._rnn_layers, batch_size, self._hid_dim).cuda())
        else:
            return weight.new_zeros(self._rnn_layers, batch_size, self._hid_dim).cuda()

    def encode_sentence(self, sentence, lengths, hidden, train=False):
        """
        simple rnn encoder.
        each token gets embedded, and calculating the rnn over all of them.
        returning the final hidden state
        """
        emb_words = self.encoder(sentence)
        packed_embd_words = pack_padded_sequence(emb_words, lengths, batch_first=True)
        if train:
            self._rnn.train() # activate dropout duirng training
        else:
            self._rnn.eval() # freeze dropout during testing

        output, hidden = self._rnn(packed_embd_words, hidden)
        if self._rnn_type == 'lstm':
            hidden = hidden[0]
        hidden = hidden.squeeze()
        return hidden

    def task_mlp(self, vec_sen, train):
        """
        calculating the mlp function over the sentence representation vector
        """

        if train:
            self.task.train()
        else:
            self.task.eval()
        return self.task(vec_sen)

    def adv_mlp(self, vec_sen, adv_ind, train, vec_drop):
        """
        calculating the adversarial mlp over the sentence representation vector.
        more than a single adversarial mlp is supported
        """
        out = vec_sen
        adv = self.advs[adv_ind]
        if train:
            if vec_drop > 0:
                vec_drop = nn.Dropout(p=vec_drop)
                out = vec_drop(vec_sen)
            self.advs[adv_ind].train()
        else:
            self.advs[adv_ind].eval()

        return self.advs[adv_ind](out)

    # based on: Unsupervised Domain Adaptation by Backpropagation, Yaroslav Ganin & Victor Lempitsky
    def calc_loss(self, sentence, lengths, y_task, y_adv, train, ro, vec_drop=0):
        """
        calculating the loss over a single example.
        accumulating the main task and adversarial task loss together.
        """
        hidden = self.init_hidden(len(sentence))
        sen = self.encode_sentence(sentence, lengths, hidden, train=train)

        task_res = self.task_mlp(sen, train)
        task_probs = F.softmax(task_res, dim=1)
        task_loss = F.cross_entropy(task_res, y_task, reduction='elementwise_mean')

        adversarial_res = []
        if ro > 0:
            adversarial_losses = []

            for i in range(self._adv_count):
                reversed_sen = ReverseLayerF.apply(sen, ro)
                adv_res = self.adv_mlp(reversed_sen, i, train, vec_drop)
                probs = F.softmax(adv_res, dim=1)
                adversarial_res.append(np.argmax(probs.cpu().detach().numpy(), axis=1))
                adversarial_losses.append(F.cross_entropy(adv_res, y_adv, reduction='elementwise_mean'))

            adversarial_loss_sum = adversarial_losses[0]
            if self._adv_count > 1:
                for i in range(1, self._adv_count):
                    adversarial_loss_sum += adversarial_losses[i]
            total_loss = task_loss + adversarial_loss_sum
        else:
            total_loss = task_loss

        return total_loss, np.argmax(task_probs.cpu().detach().numpy(), axis=1), adversarial_res

    def adv_loss(self, sentence, lengths, y_adv, train):
        """
        calculating the loss for just a single class. mainly for the baseline models.
        :param sentence:
        :param y_adv:
        :param train:
        :return:
        """
        hidden = self.init_hidden(len(sentence))
        sen = self.encode_sentence(sentence, lengths, hidden, train=train)
        adv_res = self.adv_mlp(sen, 0, train, 0)
        adv_probs = F.softmax(adv_res, dim=1)
        adv_loss = F.cross_entropy(adv_res, y_adv, reduction='elementwise_mean')
        return adv_loss, np.argmax((adv_probs.cpu().detach().numpy()), axis=1)

    #def save(self, f_name):
    #    self._model.save(f_name)

    #def load(self, f_name):
    #    self._model.populate(f_name)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
