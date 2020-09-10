# Copyright 2020 LpSS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import optimizer.core as core

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sparse_training(model, criterion, sparse_optimizer, batches, regularizer=None,
                    epochs=25, batch_size=512, knots=None, vals=None, save_path=None, show_pic=False):
    '''Training the sparse neural network with weight constrained on Lp sphere.

        training the sparse network with decreasing lp,
        the connections of each layer is adaptively adjusted based on the p and input distribution
        Args:
            model (torch.nn): the model for optimization
            criterion (torch.nn): the loss function for optimization
            sparse_optimizer (sparse_optimizer): optimizer for training sparse ANN
            batches (list of torch.utils.data): the batches for training and test
            regularizer (torch.nn.regularization,optimal): regularizer if it is possible (default: None)
            epochs (int,optional): the epoches of training (default: 25)
            batch_size (int,optional): the batch size for training (default: 512)
            knots (list of int,optional): the epoch for change learning rate (default: None)
            vals (list of float,optional): the basic learning rate in different epochs (default: None)
            save_path (str,optional): the path for saving model, and corresponding frequency (default: None)
            show_pic (Bool,optional): whether to show the training process and training data (default: False)
        '''

    train_batches, test_batches = batches
    lr = get_lr_sparse(epochs, train_batches, batch_size, knots=knots, vals=vals)
    sparse_opt = SparseOptimiser(sparse_optimizer, lr=lr)
    print('Initialization: total parameter of model is {}'.format(core.parameter_total(model)))

    return train_sparse(model, criterion, sparse_opt, train_batches, test_batches, epochs,
                        regularizer=regularizer, loggers=(core.TableLogger(),), save_path=save_path, show_pic=show_pic)


def train_sparse(model, criterion, sparse_opt, train_batches, test_batches, epochs, regularizer=None,
          loggers=(), save_path=None, test_time_in_total=True, timer=None, show_pic=None):
    timer = timer or core.Timer()
    summaries = []

    if save_path is not None:
        print('Save model: ON. Save path = {}'.format(save_path))
    if show_pic:
        show = core.plotAccu()

    for epoch in range(epochs):
        epoch_stats = train_epoch_sparse(model, criterion, train_batches, test_batches, sparse_opt, timer, regularizer,
                                  test_time_in_total=test_time_in_total)
        summary = core.union({'epoch': epoch + 1, 'lr':
            sparse_opt.param_values()['lr'] * train_batches.batch_size}, epoch_stats)
        summaries.append(summary)
        for logger in loggers:
            logger.append(summary)

        # save the model
        if save_path is not None:
            print('Saving model to the path = {}'.format(save_path))
            torch.save(model, save_path)
            print('Save finished')

        # draw the accuracy
        if show_pic:
            epoch_list = []
            test_accu = []
            for sum in summaries:
                epoch_list.append(sum['epoch'])
                test_accu.append(sum['test acc'])
            show.update(epoch_list, test_accu)

    return summaries, model


def train_epoch_sparse(model, criterion, train_batches, test_batches, sparse_optimizer, timer, regularizer=None,
                       test_time_in_total=True):
    train_stats, train_time = run_batches_sparse(model, criterion, train_batches, True, sparse_optimizer, regularizer), timer()

    test_stats, test_time = run_batches_sparse(model, criterion, test_batches, False), timer(test_time_in_total)

    para_num = core.parameter_statistics(model, 0)

    return {
        'train time': train_time, 'train loss': train_stats.mean('loss'),
        'train acc': train_stats.mean('correct'), 'para num': para_num,
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time,
    }


def run_batches_sparse(model, criterion, batches, training, optimizer=None, regularizer=None, stats=None):
    stats = stats or core.StatsLogger(('loss', 'correct'))

    for batch in batches:
        inp, target = batch
        inp = inp.to(device)
        target = target.to(device)

        if training:
            model.train()
            output = model(inp)
            output = {"loss": criterion(output, target), "correct": acc(output, target)}
            loss_out = output['loss'].sum()
            if regularizer is not None:
                loss_out = loss_out + regularizer(model)
            loss_out.backward()

            optimizer.step(criterion=criterion, inputs=inp, labels=target)
        else:
            model.eval()
            with torch.no_grad():
                output = model(inp)
            output = {"loss": criterion(output, target), "correct": acc(output, target)}

        stats.append(output)
    return stats


def acc(out, target):
    return out.max(dim=1)[1] == target


def get_lr_sparse(epochs, train_batches, batch_size, knots=None, vals=None):
    if vals is None:  # default value of vals
        vals = [0, 2, 0.01, 0.0001]
    if knots is None:
        knots = [0, epochs / 4 + 1, epochs - 4, epochs]  # default value of knots

    lr_schedule = core.PiecewiseLinear(knots, vals)
    lr = lambda step: lr_schedule(step / len(train_batches)) / batch_size
    return lr


class SparseOptimiser():
    def __init__(self, sparse_optimizer, step_number=0, **opt_params):
        self.step_number = step_number
        self.opt_params = opt_params
        self._sparse_opt = sparse_optimizer
        self._opt = sparse_optimizer._optimizer

    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k, v in self.opt_params.items()}

    def step(self, criterion=None, inputs=None, labels=None):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._sparse_opt.step(criterion, inputs, labels)

    def __repr__(self):
        return repr(self._sparse_opt)