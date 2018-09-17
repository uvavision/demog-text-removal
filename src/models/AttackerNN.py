import torch
import torch.nn as nn
import torch.nn.functionla as F

import torchvision.models as models
import torch.nn.utils

import numpy as np


class AttackerNN(nn.Module):
    def __init__(self, mlp_layers, dropout):
        super(AttackerNN, self).__init__()

        mlp = []
        for i in range(len(mlp_layers) - 1):
            mlp.append(nn.Linear(mlp_layers[i][0], mlp_layers[i][1], bias=True))
            mlp.append(nn.Dropout(p=dropout))
            mlp.append(nn.Tanh())
        mlp.append(nn.Linear(mlp_layers[-1][0], mlp_layers[-1][1], bias=True))
        self.mlp = nn.Sequential(*mlp)

        self._mlp_layers = len(mlp_layers)
        self._dropout = nn.Dropout(p=dropout)

    def calc_loss(self, enc_sen, y_adv, vec_drop, train):
        """
        the attacker core functionality.
        mlp function, with (possibely) multi layers, and at least one.
        :param enc_sen:
        :param y_adv:
        :param vec_drop:
        :param train:
        :return:
        """

        out = enc_sen
        if train:
            if vec_drop > 0:
                vec_drop = nn.Dropout(p=vec_drop)
                out = vec_drop(enc_sen)
            self.mlp.train()
        else:
            self.mlp.eval()

        out = self.mlp(out)
        task_probs = F.softmax(out)
        adv_loss = F.nll_loss(out, y_adv)
        return adv_loss, np.argmax(task_probs.data.numpy())

    #def save(self, f_name):
    #    self._model.save(f_name)

    #def load(self, f_name):
    #    self._model.populate(f_name)
