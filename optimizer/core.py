# Copyright 2020 RigL Authors.
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

import re
import time
import numpy as np
import torch
from collections import namedtuple

import matplotlib.pyplot as plt


def sub_modules(module):
    """get the sub modules of certain module, if no sub modules, return None
    Args:
        module
    Returns:
        sub_modules
    """
    return list(module.children())


def get_all_layers_with_weight(module, layer_list=None):
    """get all layers contain weight
    Args:
        module (nn.module)
        layers (list of modules)
    Returns:
        layers (list of modules)
    """
    if layer_list is None:
        layer_list = []

    if 'weight' in module._parameters:
        layer_list.append(module)
    elif hasattr(module, "weight"):
        layer_list.append(module)

    sub_layers = sub_modules(module)
    if len(sub_layers) > 0:
        for layer in sub_layers:
            layer_list = get_all_layers_with_weight(layer, layer_list)

    return layer_list


def get_weights(layer):
    """get the weight of this layer"""
    weight_list = []
    for name, param in layer.named_parameters():
        if 'weight' in name:
            weight_list.append(param)
    return weight_list


def extract_number(token):
    """Strips the number from the end of the token if it exists.

    Args:
        token: str, s or s_d where d is a number: a float or int.
        `foo_.5`, `foo_foo.5`, `foo_0.5`, `foo_4` are all valid strings.

    Returns:
    float, d if exists otherwise 1.
    """
    regexp = re.compile(r'.*_(\d*\.?\d*)$')
    if regexp.search(token):
        return float(regexp.search(token).group(1))
    else:
        return 1.


def get_n_zeros(size, sparsity):
    return int(np.ceil(sparsity * size))


def get_mask_random(shape, sparsity):
    """Creates a random sparse mask with deterministic sparsity.

    Args:
        shape: torch.Tensor, used to obtain shape of the random mask.
        sparsity: float, between 0 and 1.
        dtype: torch.dtype, type of the return value.

    Returns:
        numpy.ndarray
    """
    flat_ones = np.ones(shape).flatten()
    n_zeros = get_n_zeros(flat_ones.size, sparsity)
    flat_ones[:n_zeros] = 0

    np.random.shuffle(flat_ones)
    new_mask = flat_ones.reshape(shape)
    return new_mask


def sparsify_weight(wa, mask, h=0.1, eps=1e-8):
    '''Weight sparsification by setting the small element to zero
    Args:
        w (torch.tensor): the weight for sparsification
        mask (torch.tensor): the mask of no activated weight
        h (torch.tensor/float, optional): the threshold for sparsification (default: 0.1)

    :return: w (sparse), mask
        '''
    mask_f = mask.float()
    dim = wa.dim()
    if dim == 2:
        hh = wa.mul(mask_f).sum(1).div(mask_f.sum(1).add(eps)).mul(h)
        mask_drop = wa >= hh.view(wa.size(0), 1)
    elif dim == 3:
        hh = wa.mul(mask_f).sum(2).sum(1).div(mask_f.sum(2).sum(1).add(eps)).mul(h)
        mask_drop = wa >= hh.view(wa.size(0), 1, 1)
    elif dim == 4:
        hh = wa.mul(mask_f).sum(3).sum(2).sum(1).div(mask_f.sum(3).sum(2).sum(1).add(eps)).mul(h)
        mask_drop = wa >= hh.view(wa.size(0), 1, 1, 1)
    else:
        raise ValueError('Expected dimension of input 2 <= dims <=4, got {}'.format(dim))

    drop_n = (mask_drop ^ mask).sum()
    return mask_drop, drop_n


def sparsify_grad(ga, mask, h, eps=1e-8):
    '''grow connection by activate large gradient
    Args:
        ga (torch.tensor): the abs of gradient of weight
        mask (torch.tensor): the mask of no activated weight
        grow_n (torch.tensor/int, optional): the number of connection grows

    :return: mask
        '''
    mask_f = mask.float()
    dim = ga.dim()
    if dim == 2:
        hh = ga.mul(mask_f).sum(1).div(mask_f.sum(1).add(eps)).mul(h)
        mask_re = (ga >= hh.view(ga.size(0), 1)) | mask
    elif dim == 3:
        hh = ga.mul(mask_f).sum(2).sum(1).div(mask_f.sum(2).sum(1).add(eps)).mul(h)
        mask_re = (ga >= hh.view(ga.size(0), 1, 1)) | mask
    elif dim == 4:
        hh = ga.mul(mask_f).sum(3).sum(2).sum(1).div(mask_f.sum(3).sum(2).sum(1).add(eps)).mul(h)
        mask_re = (ga >= hh.view(ga.size(0), 1, 1, 1)) | mask
    else:
        raise ValueError('Expected dimension of input 2 <= dims <=4, got {}'.format(dim))

    return mask_re, mask_re ^ mask


def weight_statstics(model, eps=1e-8):
    layers = get_all_layers_with_weight(model)
    f = 0
    for layer in layers:
        if 'weight' in layer._parameters:
            f += (layer._parameters['weight'].data.abs() <= eps).sum().item()
        elif hasattr(layer, "weight"):
            f += (layer.weight.data.abs() <= eps).sum().item()
    return f


def parameter_statistics(model, eps=1e-8):
    return parameter_total(model) - weight_statstics(model, eps)


def parameter_total(model):
    f = 0
    for p in model.parameters():
        f += np.prod(p.size()).item()
    return f


def sparsity(model, eps=1e-8):
    return weight_statstics(model, eps) / parameter_total(model)


def cat(*xs):
    return torch.cat(xs)


def to_numpy(x):
    return x.detach().cpu().numpy()


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class StatsLogger():
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)


class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered))


class StatsLogger():
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)


class plotAccu():
    def __init__(self):
        self.plt_h = None

    def update(self, xs, ys, legends=None, title=None, xlabel=None, ylabel=None):
        n = len(xs)
        m = len(ys)

        if m != n:
            ValueError('inputs: x, y should have the same length, got {}, {}'.format(n, m))

        if self.plt_h is None:
            self.plt_h = []
            max_x = 0
            for i, x in enumerate(xs):
                self.plt_h.append(plt.plot(x, ys[i]))
                max_x = np.max([max_x] + x)

            plt.xlim([0, max_x])
            plt.ylim([0, 1])

            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            if legends is not None:
                plt.legend(labels=legends)
            if title is not None:
                plt.title(title)

            plt.pause(0.001)
        else:
            max_x = 0
            for i, x in enumerate(xs):
                self.plt_h[i][0].set_xdata(x[i])
                self.plt_h[i][0].set_ydata(ys[i])
                max_x = np.max([max_x] + x)

            plt.xlim([0, max_x])
            plt.pause(0.0001)