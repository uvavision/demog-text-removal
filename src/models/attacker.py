"""
Usage:
  sent_demog_attacker.y [--batch_size=NUM]
  [--epochs=EPOCHS] [--ro=RO] [--model=MODEL] [--task=TASK] [--num_adv=NUM_ADV] [--adv_depth=NUM_HID_LAY]
  [--use_s] [--mlp_layers=NUM_LAY] [--vec_drop=DROPOUT] [--lr=LR]
  [--lstm_size=LSTM_SIZE] [--dropout=DROPOUT] [--enc_size=ENC_SIZE] [--rnn_type=RNN_TYPE] [--data=DATA]
  [--adv_size=ADV_SIZE] [--att_hid_size=ATT_HID_MLP] [--init=INIT]
  [--model_flip=MODEL_FLIP] [--main_task]


Options:
  -h --help                     show this help message and exit
  --batch_size=NUM              batch size
  --epochs=EPOCHS               amount of training epochs [default: 100]
  --ro=RO                       amount of power to the adversarial
  --model=MODEL                 model name
  --task=TASK                   name of task [default: sentiment]
  --num_adv=NUM_ADV             number of simultaneous adversarials trainer [default: 1]
  --adv_depth=NUM_HID_LAY       encoders' adversarial depth [default: 1]
  --use_s                       use feature s which we want to be invariant to
  --mlp_layers=NUM_LAY          number of layers of mlp [default: 2]
  --vec_drop=DROPOUT            dropout on the representation vector [default: 0]
  --lstm_size=LSTM_SIZE         size of lstm layer [default: 1]
  --dropout=DROPOUT             dropout probability [default: 0.2]
  --lr=LR                       learning rate [default: 0.01]
  --enc_size=ENC_SIZE           size of lstm hidden layer [default: 300]
  --rnn_type=RNN_TYPE           type of rnn - lstm/gru [default: lstm]
  --data=DATA                   data source [default: normal]
  --adv_size=ADV_SIZE           size of adversarial hidden layer [default: 300]
  --att_hid_size=ATT_HID_MLP    size of adversarial mlp hidden layer [default: 300]
  --init=INIT                   init value [default: 0]
  --model_flip=MODEL_FLIP       switch models parameters [default: 0]
  --main_task                   train for main task

"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import numpy as np
import shutil
from docopt import docopt
from sklearn.metrics import accuracy_score
from tensorboard_logger import configure, log_value
from os.path import expanduser

from AdvNN import AdvNN
from AttackerNN import AttackerNN
from data_handler import get_data
from training_utils import get_logger, task_dic
from consts import SEED, models_dir, tensorboard_dir, data_dir

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def epoch_pass(data_loader, model, enc_net, optimizer, training, batch_size, vec_drop, truth_ind, \
        logger, print_every=20000):

    t_loss = 0.0
    preds, truth = [], []
    n_processed = 0

    for ind, row in enumerate(data):
        sent = enc_net.encode_sentence(row[0].cuda(), train=False)
        loss, adv_pred = model.calc_loss(sent, row[truth_ind], vec_drop, train=training)
        preds.append(adv_pred.cpu().numpy())
        truth.append(row[truth_ind].numpy())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(row)

        if (ind + 1) % print_every == 0:
            logger.debug('{0}: task loss: {1}, adv acc: {2}'.format(ind + 1, t_loss / n_processed, \
                    accuracy_score(truth, preds)))
    return accuracy_score(truth, preds), t_loss / n_processed


def train(model, enc_net, train_loader, dev_loader, trainer, epochs, vec_drop, batch_size, \
        logger, truth_ind=2, print_every=20000):
    """
    training method
    :param model: attacker model
    :param train: training set
    :param dev: development/test set
    :param trainer: optimizer
    :param epochs: number of epochs
    :param vec_drop: representation vector (of the sentence) dropout
    :param batch_size: size of batch
    :param logger:
    :param truth_ind: index of the truth in the train/dev set
    :param print_every: print every x examples in each epoch
    :return:
    """
    train_acc_arr, train_loss_arr = [], []
    dev_acc_arr, dev_loss_arr = [], []
    best_model_epoch = 1
    best_score = 0.0

    logger.debug('training started')
    for epoch in xrange(1, epochs + 1):

        # train
        epoch_pass(train_loader, model, enc_net, optimizer, True, batch_size, vec_drop, truth_ind, \
                logger, print_every)
        train_task_acc, loss = epoch_pass(train_loader, model, enc_net, optimizer, False, batch_size, \
                vec_drop, truth_ind, logger, print_every)
        train_acc_arr.append(train_task_acc)
        train_loss_arr.append(loss)
        logger.debug('train, {0}, adv acc: {1}'.format(epoch, train_task_acc))

        # dev
        dev_task_acc, loss = epoch_pass(dev_loader, model, enc_net, optimizer, False, batch_size, \
                vec_drop, truth_ind, logger, print_every)
        dev_acc_arr.append(dev_task_acc)
        dev_loss_arr.append(loss)

        logger.debug('dev, {0}, adv acc: {1}'.format(epoch, dev_task_acc))
        log_value('attacker-acc', dev_task_acc, epoch)

        if dev_task_acc > best_score:
            best_score = dev_task_acc
            best_model_epoch = epoch
            torch.save(model.state_dict(), models_dir + task + '/best_attacker')
    logger.info('best_score:' + str(best_score))
    logger.info('best_epoch:' + str(best_model_epoch))
    logger.info('train_task_acc:' + str(train_acc_arr))
    logger.info('train_loss:' + str(train_loss_arr))
    logger.info('dev_task_acc:' + str(dev_acc_arr))
    logger.info('dev_loss:' + str(dev_loss_arr))


def text2vectors(enc_model, data):
    """
    convert all text to vectors using the encoder from the training phase
    :param enc_model: encoder model
    :param data: textual data (in list) to convert to vectors
    :return: list of vectors
    """
    data_vectors = []
    for ind, row in enumerate(data):
        sent = enc_model.encode_sentence(row[0], train=False)
        data_vectors.append((sent.npvalue(), row[1], row[2]))
    return data_vectors


if __name__ == '__main__':
    arguments = docopt(__doc__)

    task_str = arguments['--task']
    num_adv = int(arguments['--num_adv'])
    base_model = arguments['--model']
    task = 'attacker-' + base_model
    batch_size = int(arguments['--batch_size'])

    if 'unseen' in task_str:
        task += '-data:' + task_str
    elif task_str not in base_model:
        task += '-task:' + task_str

    mlp_lay = int(arguments['--mlp_layers'])
    if mlp_lay != 2:
        task += '-mlp_lay:' + str(mlp_lay)

    att_mlp_size = int(arguments['--att_hid_size'])
    if att_mlp_size != 300:
        task += '-att_mlp:' + str(att_mlp_size)

    hid_size = int(arguments['--enc_size'])

    w_s = bool(arguments['--use_s'])
    lstm_size = int(arguments['--lstm_size'])
    dropout = float(arguments['--dropout'])
    if dropout != 0.2:
        task += '-dropout:' + str(dropout)

    adv_depth = int(arguments['--adv_depth'])

    vec_drop = float(arguments['--vec_drop'])
    if vec_drop != 0:
        task += '-rep_dropout:' + str(vec_drop)

    if arguments['--model_flip'] != '0':
        task += '-flip_comp:' + arguments['--model_flip']
        if arguments['--main_task']:
            task += '-main_task'

    init = arguments['--init']
    if int(init) != 0:
        task += '-' + init

    attacked_model = base_model
    print 'attacking: {0}'.format(attacked_model)

    adv_hid_size = int(arguments['--adv_size'])

    home = expanduser("~")

    input_dir = data_dir

    input_dir += task_dic[task_str] + '/'

    input_vocab = input_dir + 'vocab'

    model_dir = models_dir + task
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print 'already exist. exiting...'
        exit(-1)
    logger = get_logger('logging', model_dir)
    logger.info(arguments)
    logger.info(task)

    configure(tensorboard_dir + task)
    train_data, test_data = get_data(task_str, input_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    with open(input_vocab, 'r') as f:
        vocab = f.readlines()
        vocab = map(lambda s: s.strip(), vocab)
    vocab_size = len(vocab)
    enc_net = AdvNN(hid_size, hid_size, 2, hid_size, adv_hid_size, 2, num_adv, vocab_size, dropout,
                    lstm_size, adv_depth, rnn_type=arguments['--rnn_type'])
    enc_net = nn.DataParallel(enc_net).cuda()
    checkpoint = torch.load(models_dir + attacked_model)
    enc_net.load_state_dict(checkpoint['state_dict'])

    if arguments['--model_flip'] != '0':
        encoder2 = AdvNN(hid_size, hid_size, 2, hid_size, adv_hid_size, 2, 5, vocab_size, dropout,
                         lstm_size, adv_depth)
        encoder2.load(models_dir + 'sent_race-n_adv:5-ro:1.0/epoch_50')
        if arguments['--model_flip'] == 'emb':
            enc_net.encoder = encoder2.encoder
        else:
            enc_net._rnn = encoder2._rnn

    # x_train = text2vectors(enc_net, x_train)
    # x_test = text2vectors(enc_net, x_test)

    mlp = []
    for i in range(mlp_lay - 1):
        mlp.append((hid_size, att_mlp_size))
    if mlp_lay == 1:
        mlp.append((hid_size, 2))
    else:
        mlp.append((att_mlp_size, 2))
    adv_net = AttackerNN(mlp, dropout)

    lr = float(arguments['--lr'])
    if 'fix' in task or 'length' in task:
        lr = 0.001

    optimizer = optim.SGD(adv_net.parameters(), lr=lr, momentum=0.9)

    if arguments['--main_task']:
        m_task = 1
    else:

        m_task = 2
    train(adv_net, enc_net, train_loader, test_loader, optimizer, int(arguments['--epochs']), \
            vec_drop, batch_size, logger, m_task)
