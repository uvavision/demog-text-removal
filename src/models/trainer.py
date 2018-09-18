"""
Usage:
  sent_demog_train.py [--batch_size=NUM]
  [--epochs=EPOCHS] [--ro=RO] [--task=TASK] [--type=TYPE] [--num_adv=NUM_ADV] [--adv_depth=NUM_HID_LAY]
  [--lstm_size=LSTM_SIZE] [--dropout=DROPOUT] [--rnn_dropout=RNN_DROPOUT] [--rnn_type=RNN_TYPE]
  [--enc_size=ENC_SIZE] [--adv_size=ADV_SIZE] [--vec_drop=DROPOUT] [--init=INIT]


Options:
  -h --help                     show this help message and exit
  --batch_size=NUM              batch size
  --epochs=EPOCHS               amount of training epochs [default: 100]
  --ro=RO                       amount of power to the adversarial
  --task=TASK                   single task to train [default: sentiment]
  --type=TYPE                   type of the task (1/2) [default: 1]
  --num_adv=NUM_ADV             number of simultaneous adversarials trainer [default: 1]
  --adv_depth=NUM_HID_LAY       number of hidden layers of adversarials [default: 1]
  --lstm_size=LSTM_SIZE         size of lstm layer [default: 1]
  --dropout=DROPOUT             dropout probability [default: 0.2]
  --rnn_dropout=RNN_DROPOUT     rnn dropout [default: 0.0]
  --rnn_type=RNN_TYPE           type of rnn - lstm/gru [default: lstm]
  --enc_size=ENC_SIZE           size of lstm hidden layer [default: 300]
  --adv_size=ADV_SIZE           size of adversarial hidden layer [default: 300]
  --vec_drop=DROPOUT            dropout on the representation vector [default: 0]
  --init=INIT                   init value [default: 0]

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
from consts import SEED, data_dir, models_dir, tensorboard_dir
from data_handler import get_data, collate_fn
from training_utils import get_logger, task_dic

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def epoch_pass(data_loader, model, optimizer, training, ro, vec_drop, logger, print_every=500):
    """
    run a single epoch pass on the data
    :param data: data to train/predict on
    :param model: .
    :param trainer: optimizer
    :param training: boolean - for training/testing purposes
    :param ro: lambda used in paper. determines the adversarial power
    :param vec_drop: dropout on the representation vector
    :param logger: .
    :param print_every: print the accumulative stats
    :return: accuracy score of the main task, the adversary task and the total loss.
    """
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    n_processed = 0
    t_loss = 0.0
    for i in range(num_adv):
        adv_preds.append([])

    for ind, data in enumerate(data_loader):
        loss, task_pred, adv_pred = model.calc_loss(data[0].cuda(), data[3], data[1].cuda(), \
                data[2].cuda(), training, \
                ro, vec_drop)
        task_preds += task_pred.tolist()
        task_truth += data[1].numpy().tolist()

        for i in range(len(adv_pred)):
            adv_preds[i] += adv_pred[i].tolist()

        adv_truth += data[2].numpy().tolist()

        t_loss += loss.item()
        n_processed += len(data)

        if training:
            # backpropogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (ind + 1) % print_every == 0:
            mean = 0.0
            for i in range(num_adv):
                mean += accuracy_score(adv_truth, adv_preds[i])
            mean /= len(adv_pred)
            logger.debug('{0}: task loss: {1}, task acc: {2}, mean adv acc: {3}'
                         .format(ind + 1, t_loss / n_processed,
                                 accuracy_score(task_truth, task_preds), mean))
    adv_res = []
    for i in range(num_adv):
        adv_res.append(accuracy_score(adv_truth, adv_preds[i]))
    return accuracy_score(task_truth, task_preds), adv_res, t_loss / n_processed

def train(model, train_loader, dev_loader, optimizer, epochs, \
        vec_drop, logger, print_every=500):
    """
    the training function with the adversarial usage
    """
    train_task_acc_arr, train_adv_acc_arr, train_loss_arr = [], [], []
    dev_task_acc_arr, dev_adv_acc_arr, dev_loss_arr = [], [], []
    best_score = 0.0

    ro = float(arguments['--ro'])
    logger.debug('training started')
    for epoch in xrange(1, epochs + 1):

        # train
        epoch_pass(train_loader, model, optimizer, True, ro, vec_drop, \
                logger, print_every)
        train_task_acc, train_adv_acc, loss = epoch_pass(train_loader, model, optimizer, False, ro, \
                vec_drop, logger, print_every)

        train_task_acc_arr.append(train_task_acc)
        train_adv_acc_arr.append(train_adv_acc)
        train_loss_arr.append(loss)
        logger.debug('train, {0}, {1}, {2}'.format(epoch, train_task_acc, train_adv_acc))

        # dev
        dev_task_acc, dev_adv_acc, loss = epoch_pass(dev_loader, model, optimizer, False, ro, \
                0, logger, print_every)
        dev_task_acc_arr.append(dev_task_acc)
        dev_adv_acc_arr.append(dev_adv_acc)
        dev_loss_arr.append(loss)
        logger.debug('dev, {0}, {1}, {2}'.format(epoch, dev_task_acc, np.mean(dev_adv_acc)))
        log_value('dev-task-acc', dev_task_acc, epoch)
        log_value('dev-mean-adv-acc', np.mean(dev_adv_acc), epoch)

        is_best = dev_adv_acc > best_score
        best_score = max(dev_adv_acc, best_score)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_score': best_score}, is_best, models_dir + task + '/checkpoint.pth.tar')

    logger.info('best_score:' + str(best_score))
    logger.info('train_task_acc:' + str(train_task_acc_arr))
    logger.info('train_adv_acc:' + str(train_adv_acc_arr))
    logger.info('train_loss:' + str(train_loss_arr))
    logger.info('dev_task_acc:' + str(dev_task_acc_arr))
    logger.info('dev_adv_acc:' + str(dev_adv_acc_arr))
    logger.info('dev_loss:' + str(dev_loss_arr))


def train_task(model, train_loader, dev_loader, optimizer, epochs, task_type, \
        logger, print_every=500):
    """
    the training function for a single task. used for the baseline experiments
    """
    # train_task_acc_arr, train_loss_arr = [], []
    dev_task_acc_arr, dev_loss_arr = [], []
    best_score = 0.0

    logger.debug('training started')
    for epoch in xrange(1, epochs + 1):
        t_loss = 0.0
        adv_preds, adv_truth = [], []
        n_processed = 0
        # train
        for ind, data in enumerate(train_loader):
            loss, adv_pred = model.adv_loss(data[0].cuda(), data[3], data[task_type].cuda(), \
                    True)
            adv_preds += adv_pred.tolist()
            adv_truth += data[task_type].numpy().tolist()
            t_loss += loss.item()
            n_processed += len(data)

            # backpropogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (ind + 1) % print_every == 0:
                logger.debug('{0}: task loss: {1}, adv acc: {2}'.format(ind + 1, t_loss / n_processed, \
                        accuracy_score(adv_truth, adv_preds)))
        # dev
        t_loss = 0.0
        adv_preds = []
        adv_truth = []
        n_processed = 0
        for ind, data in enumerate(dev_loader):
            loss, adv_pred = model.adv_loss(data[0].cuda(), data[3], data[task_type].cuda(), \
                    False)
            adv_preds += adv_pred.tolist()
            adv_truth += data[task_type].numpy().tolist()

            t_loss += loss.item()
            n_processed += len(data)

        adv_acc = accuracy_score(adv_truth, adv_preds)
        dev_task_acc_arr.append(adv_acc)
        dev_loss_arr.append(t_loss / n_processed)
        logger.debug('dev-task epoch: {0}, acc: {1}, loss: {2}'.format(epoch, adv_acc, \
                t_loss / n_processed))
        log_value('dev-task-acc', adv_acc, epoch)
        is_best = adv_acc > best_score
        best_score = max(best_score, adv_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_score': best_score}, is_best, models_dir + task + '/checkpoint.pth.tar')
    logger.info('best_score:' + str(best_score))
    logger.info('dev_task_acc:' + str(dev_task_acc_arr))
    logger.info('dev_loss:' + str(dev_loss_arr))

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint', 'model_best'))

if __name__ == '__main__':
    arguments = docopt(__doc__)

    ro = arguments['--ro']
    num_adv = int(arguments['--num_adv'])
    lstm_size = int(arguments['--lstm_size'])
    task_str = arguments['--task']
    num_epoch = int(arguments['--epochs'])
    batch_size = int(arguments['--batch_size'])

    input_dir = data_dir

    if task_str not in task_dic:
        print 'task not supported in task_dic'
        exit(-1)
    input_dir += task_dic[task_str] + '/'
    input_vocab = input_dir + 'vocab'

    task = task_str
    task_type = int(arguments['--type'])

    if ro != str(-1):
        print 'using adversarial'
        task += '-n_adv:' + str(num_adv)
    else:
        if task_type == 1:
            task += '-type:1'
        else:
            task += '-type:2'

    pre_w2id = None

    if lstm_size > 1:
        task += '-lstm_size:{0}'.format(lstm_size)

    hid_size = int(arguments['--enc_size'])
    if hid_size != 300:
        task += '-hid:' + str(hid_size)

    rnn_type = arguments['--rnn_type']
    if rnn_type != 'lstm':
        task += '-' + arguments['--rnn_type']

    if ro != -1.0 and ro != 1.0:
        task += '-ro:' + str(ro)

    dropout = float(arguments['--dropout'])
    if dropout != 0.2:
        task += '-dropout:' + str(dropout)
        print 'dropout: {0}'.format(dropout)

    rnn_dropout = float(arguments['--rnn_dropout'])
    if rnn_dropout != 0.0:
        task += 'rnn_dropout:' + str(rnn_dropout)
        print 'using rnn dropout: {0}'.format(rnn_dropout)

    adv_hid_size = int(arguments['--adv_size'])
    if adv_hid_size != 300:
        task += '-adv_hid_size:' + str(adv_hid_size)
        print 'adversary hidden size: {0}'.format(adv_hid_size)

    adv_depth = int(arguments['--adv_depth'])
    if adv_depth != 1:
        task += '-depth:' + str(str(adv_depth))
        print 'adversary depth: {0}'.format(adv_depth)

    vec_dropout = float(arguments['--vec_drop'])
    if vec_dropout != 0.0:
        task += '-rep_dropout:' + str(vec_dropout)
        print 'dropout on the sentence representation'

    init = arguments['--init']
    if int(init) != 0:
        task += '-' + init
    model_dir = models_dir + task
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        print 'already exist. exiting...'
        exit(-1)

    logger = get_logger(task, model_dir)

    logger.info(arguments)
    logger.info(task)
    home = expanduser("~")
    configure(tensorboard_dir + task)

    train_data, test_data = get_data(task_str, input_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, \
            num_workers=6, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, \
            num_workers=4, collate_fn=collate_fn)

    out_size = 2
    with open(input_vocab, 'r') as f:
        vocab = f.readlines()
        vocab = map(lambda s: s.strip(), vocab)
    vocab_size = len(vocab)
    adv_net = AdvNN(hid_size, hid_size, out_size, hid_size, adv_hid_size, out_size, num_adv, vocab_size,
                    dropout, lstm_size, adv_depth, rnn_dropout=rnn_dropout, rnn_type=rnn_type)
    adv_net = adv_net.cuda()

    optimizer = optim.SGD(adv_net.parameters(), lr=0.01, momentum=0.9)

    if ro == str(-1):
        logger.debug('1 task')
        train_task(adv_net, train_loader, test_loader, optimizer, num_epoch, task_type, logger)
    else:
        logger.debug('2 tasks')
        train(adv_net, train_loader, test_loader, optimizer, num_epoch, vec_dropout, logger)
