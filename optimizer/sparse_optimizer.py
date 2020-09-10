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

"""This module implements some common and new sparse training algorithms.
Pytorch version of SET, RigL, Static and LpSS
"""

import torch
import torch.nn.utils.prune as prune
import numpy as np
import optimizer.core as co
from optimizer.core import (get_all_layers_with_weight, extract_number, get_mask_random, sparsify_weight, sparsity)

import optimizer.myFunction as mF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimizer(object):
    """Implementation of optimizer.

        Basic class of optimizer.
        This optimizer wraps a regular optimizer and performs static masks according to the default sparsity.

        Attributes:
            model: torch.nn
            optimizer: torch.Optimizer
            default_sparsity: float, the sparsity of nn
            mask_init_method: str, of method for initialization
    """
    def __init__(self, model, optimizer, default_sparsity=0.3, mask_init_method='random'):
        super(Optimizer, self).__init__()

        self._optimizer = optimizer
        self._default_sparsity = default_sparsity
        self._mask_init_method = mask_init_method

    def step(self, criterion=None, inputs=None, labels=None):
        # update parameters
        self._optimizer.step()
        self._optimizer.zero_grad()


class SparseSETOptimizer(Optimizer):
    """Implementation of dynamic sparsity optimizers.

        Implementation of SET.
        See https://www.nature.com/articles/s41467-018-04316-3
        This optimizer wraps a regular optimizer and performs updates on the masks
        according to schedule given.

        Attributes:
            model: torch.nn
            optimizer: torch.Optimizer
            begin_epoch: float, first iteration where masks are updated.
            end_epoch: float, iteration after which no mask is updated.
            train_loader: dataset, contain samples of each batch
            frequency: int, of mask update operations.
            default_sparsity: float, the sparsity of nn
            mask_init_method: str, of method for initialization
            drop_fraction: float, of connections to drop during each update.
            drop_fraction_anneal: str or None, if supplied used to anneal the drop
              fraction.
            grow_init: str, name of the method used to initialize new connections.
    """

    def __init__(self, model, optimizer, begin_epoch, end_epoch, train_loader, frequency, default_sparsity=0.3,
                 mask_init_method='random', drop_fraction=0.1, drop_fraction_anneal='constant', grow_init='zeros'):
        super(SparseSETOptimizer, self).__init__(
            model, optimizer, default_sparsity=default_sparsity, mask_init_method=mask_init_method)

        self._grow_init = grow_init
        self._drop_fraction_anneal = drop_fraction_anneal
        self._drop_fraction_initial_value = drop_fraction
        self._drop_fraction = drop_fraction
        self._begin_epoch = begin_epoch
        self._end_epoch = end_epoch
        self._batch_epo = len(train_loader)
        self._frequency = frequency
        self._frequency_val = frequency
        self.global_step = 0

        # get the names and parameters
        layers = get_all_layers_with_weight(model)
        masks = []
        for i, layer in enumerate(layers):
            while(True):
                layer = layers[i]
                layer_name = layer.__class__.__name__
                if ('Norm' in layer_name):
                    # remove the normalization layers
                    layers.remove(layer)
                else:
                    # only the weight in convolution or full connected layers should be sparsified
                    weight = layer.weight
                    masks.append(torch.ones(weight.size(), dtype=torch.bool, device=device))
                    break

        self.layers = layers
        self.masks = masks
        self.setup_graph()

        print('initial sparsity = {}'.format(sparsity(model, 0)))

    def setup_graph(self):
        # initialize the masks
        if self._mask_init_method == 'random':
            for i, mask in enumerate(self.masks):
                size = mask.size()
                self.masks[i] = torch.tensor(get_mask_random(size, self._default_sparsity), dtype=mask.dtype).to(device)

        # initialize masked weight
        for i, layer in enumerate(self.layers):
            if prune.is_pruned(layer):
                prune.remove(layer, 'weight')
            prune.custom_from_mask(layer, 'weight', self.masks[i])

    def step(self, criterion=None, inputs=None, labels=None):
        # update parameters
        self._optimizer.step()
        self._optimizer.zero_grad()

        # update fraction
        if self._begin_epoch <= self.global_step/self._batch_epo < self._end_epoch and \
                self.global_step % self._frequency == 0:
            self.update_topology()

        self.global_step += 1

    def update_topology(self):
        # update fraction
        self.update_drop_fraction()
        masks = self.masks
        new_masks = []

        # update topology
        for i, layer in enumerate(self.layers):
            masks[i], new_mask = self.update_layer_mask(layer, masks[i])
            new_masks.append(new_mask)

        # clear the masked momentum gradient
        self.reset_momentum(masks, new_masks)

    def update_drop_fraction(self):
        """Returns a constant or annealing drop_fraction op."""
        cur_epo = self.global_step/self._batch_epo
        if self._drop_fraction_anneal == 'constant':
            self._drop_fraction = self._drop_fraction_initial_value
        elif self._drop_fraction_anneal == 'cosine':
            decay_epochs = self._end_epoch - self._begin_epoch
            self._drop_fraction = self._drop_fraction_initial_value * (np.cos(np.pi*(cur_epo-self._begin_epoch)/decay_epochs)*0.5+0.5)
        elif self._drop_fraction_anneal.startswith('exponential'):
            decay_epochs = self._end_epoch - self._begin_epoch
            exponent = extract_number(self._drop_fraction_anneal)
            self._drop_fraction = self._drop_fraction_initial_value * np.exp(-exponent*(cur_epo-self._begin_epoch)/decay_epochs)
        else:
            raise ValueError('drop_fraction_anneal: %s is not valid' %
                             self._drop_fraction_anneal)

    def update_layer_mask(self, layer, layer_mask, noise_std=1e-5):
        layer_weight = layer.weight

        # Add noise for slight bit of randomness and drop
        masked_weights = layer_mask * layer_weight
        score_drop = masked_weights.abs() + self._random_normal(layer_weight.size(), std=noise_std)
        layer_mask_dropped, n_prune = self.drop_minimum(score_drop, layer_mask)

        # Randomly revive n_prune many connections from non-existing connections.
        score_grow = self._random_uniform(layer_weight.size()) * (~layer_mask_dropped)
        layer_mask, new_mask = self.grow_maximum(score_grow, layer_mask_dropped, n_prune)

        # update the weight
        if prune.is_pruned(layer):
            prune.remove(layer, 'weight')
        prune.custom_from_mask(layer, 'weight', layer_mask)

        return layer_mask, new_mask

    def drop_minimum(self, layer_score, layer_mask):
        """drop the weights with minimum score (weight or the gradient of weight)"""
        n_ones = int(layer_mask.sum())
        n_prune = int(self._drop_fraction*n_ones)
        n_keep = n_ones - n_prune

        _, indices = layer_score.view([-1]).sort(descending=True)
        indices = indices[:n_keep]

        layer_mask_drop = torch.zeros(layer_mask.size(), dtype=layer_mask.dtype).to(device)
        mask_vec = layer_mask_drop.view([-1])
        mask_vec[indices] = True

        return layer_mask_drop, n_prune

    def grow_maximum(self, layer_score, layer_mask, n_prune):
        """grow the weights with maximum score (weight or the gradient of weight)"""
        _, indices = layer_score.view([-1]).sort(descending=True)
        indices = indices[:n_prune]

        mask_vec = layer_mask.view([-1])
        mask_vec[indices] = True

        new_mask = torch.zeros(layer_mask.size(), dtype=layer_mask.dtype).to(device)
        new_mask.view([-1])[indices] = True

        return layer_mask, new_mask

    def reset_momentum(self, masks, new_masks):
        optimizer = self._optimizer
        group = optimizer.param_groups[0]
        momentum = group['momentum']
        if momentum != 0:
            for i, layer in enumerate(self.layers):
                weight_orig = layer.weight_orig
                param_state = optimizer.state[weight_orig]
                param_state['momentum_buffer'].mul_(masks[i])
                param_state['momentum_buffer'].mul_(~new_masks[i])

    def get_grow_tensor(self, weights, method):
        """Different ways to initialize new connections.

        Args:
          weights: torch.Tensor or Variable.
          method: str, available options: 'zeros', 'random_normal', 'random_uniform'
            and 'initial_value'

        Returns:
          torch.Tensor same shape and type as weights.

        Raises:
          ValueError, when the method is not valid.
        """
        if not isinstance(method, str):
            raise ValueError('Grow-Init: {} is not a string'.format(method))

        if method == 'zeros':
            grow_tensor = torch.zeros(weights, dtype=weights.dtype)
        elif method.startswith('initial_dist'):
            original_size = weights.size()
            divisor = extract_number(method)
            grow_tensor = torch.reshape(
                weights.view([-1])[torch.randperm(len(weights.view([-1])))], original_size) / divisor
        elif method.startswith('random_normal'):
            original_size = weights.size()
            std = weights.std()
            divisor = extract_number(method)
            grow_tensor = self._random_normal(
                original_size, std=std, dtype=weights.dtype) / divisor
        elif method.startswith('random_uniform'):
            original_size = weights.size()
            mean = weights.abs().mean()
            divisor = extract_number(method)
            grow_tensor = self._random_uniform(
                original_size, minval=-mean, maxval=mean, dtype=weights.dtype) / divisor
        else:
            raise ValueError('Grow-Init: %s is not a valid option.' % method)
        return grow_tensor

    def _random_uniform(self, size, minval=None, maxval=None, dtype=None):
        if minval and maxval:
            return torch.rand(size, dtype=dtype).mul(maxval-minval).add(minval).to(device)
        else:
            return torch.rand(size, dtype=dtype).to(device)

    def _random_normal(self, size, std, dtype=None):
        return torch.randn(size, dtype=dtype).mul(std).to(device)


class SparseStaticOptimizer(SparseSETOptimizer):
    """Sparse optimizer that re-initializes weak connections during training.

    Attributes:
        model: torch.nn
        optimizer: torch.Optimizer
        begin_epoch: float, first iteration where masks are updated.
        end_epoch: float, iteration after which no mask is updated.
        train_loader: dataset, contain samples of each batch.
        frequency: int, of mask update operations.
        default_sparsity: float, the sparsity of nn
        mask_init_method: str, of method for initialization
        drop_fraction: float, of connections to drop during each update.
        drop_fraction_anneal: str or None, if supplied used to anneal the drop
          fraction.
        grow_init: str, name of the method used to initialize new connections.
     """

    def __init__(self, model, optimizer, begin_epoch, end_epoch, train_loader, frequency, default_sparsity=0.3,
                 mask_init_method='random', drop_fraction=0.1, drop_fraction_anneal='constant', grow_init='zeros'):
        super(SparseStaticOptimizer, self).__init__(
            model, optimizer, begin_epoch, end_epoch, train_loader, frequency,
            default_sparsity=default_sparsity, mask_init_method=mask_init_method,
            drop_fraction=drop_fraction, drop_fraction_anneal=drop_fraction_anneal,
            grow_init=grow_init)

    def update_layer_mask(self, layer, layer_mask, noise_std=1e-5):
        layer_weight = layer.weight

        # Add noise for slight bit of randomness and drop
        masked_weights = layer_mask * layer_weight
        score_drop = masked_weights.abs() + self._random_normal(layer_weight.size(), std=noise_std)
        layer_mask_dropped, n_prune = self.drop_minimum(score_drop, layer_mask)

        # Randomly revive n_prune many connections from non-existing connections.
        score_grow = (layer_mask * (~layer_mask_dropped)).float()
        layer_mask, new_mask = self.grow_maximum(score_grow, layer_mask_dropped, n_prune)

        # update the weight
        if prune.is_pruned(layer):
            prune.remove(layer, 'weight')
        prune.custom_from_mask(layer, 'weight', layer_mask)

        return layer_mask, new_mask


class SparseRigLOptimizer(SparseSETOptimizer):
    """Sparse optimizer that grows connections with the pre-removal gradients.
    Implementation of RigL
    https://arxiv.org/abs/1911.11134
    Attributes:
        model: torch.nn
        optimizer: torch.Optimizer
        begin_epoch: float, first iteration where masks are updated.
        end_epoch: float, iteration after which no mask is updated.
        train_loader: dataset, contain samples of each batch.
        frequency: int, of mask update operations.
        drop_fraction: float, of connections to drop during each update.
        drop_fraction_anneal: str or None, if supplied used to anneal the drop
          fraction.
        grow_init: str, name of the method used to initialize new connections.
        initial_acc_scale: float, used to scale the gradient when initializing the,
          momentum values of new connections. We hope this will improve training,
          compare to starting from 0 for the new connections. Set this to something
          between 0 and 1 / (1 - momentum). This is because in the current
          implementation of MomentumOptimizer, aggregated values converge to
          1 / (1 - momentum) with constant gradients.
    """

    def __init__(self, model, optimizer, begin_epoch, end_epoch, train_loader, frequency=50, default_sparsity=0.3,
                 drop_fraction=0.1, drop_fraction_anneal='constant',
                 grow_init='zeros', initial_acc_scale=0.):
        super(SparseRigLOptimizer, self).__init__(
            model, optimizer, begin_epoch, end_epoch, train_loader, frequency,
            default_sparsity=default_sparsity,
            drop_fraction=drop_fraction, drop_fraction_anneal=drop_fraction_anneal,
            grow_init=grow_init)
        self._initial_acc_scale = initial_acc_scale
        self.model = model
        self.grad_dict = None

    def step(self, criterion=None, inputs=None, labels=None):
        # update parameters
        self._optimizer.step()
        self._optimizer.zero_grad()

        # print('nonzero parameters = {}'.format(co.parameter_statistics(self.model, 0)))

        # update fraction
        if self._begin_epoch <= self.global_step / self._batch_epo < self._end_epoch and \
                self.global_step % self._frequency == 0:
            self.compute_gradients(criterion, inputs, labels)  # update the gradient for grow connections
            self.update_topology()
            self._optimizer.zero_grad()

        self.global_step += 1

    def compute_gradients(self, criterion, inputs, labels):
        """Wraps the compute gradient of passed optimizer."""
        # remove the prune and forward
        for layer in self.layers:
            if prune.is_pruned(layer):
                prune.remove(layer, 'weight')

        # forward
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss = loss.sum()

        # backward and calculate all gradients
        loss.backward()

        # get gradient
        gradient = []
        for layer in self.layers:
            gradient.append(layer.weight.grad.data)

        self.grad_dict = dict(zip(self.layers, gradient))

    def update_layer_mask(self, layer, layer_mask, noise_std=1e-5):
        layer_weight = layer.weight

        # Add noise for slight bit of randomness and drop
        masked_weights = layer_mask * layer_weight
        score_drop = masked_weights.abs() + self._random_normal(layer_weight.size(), std=noise_std)
        layer_mask_dropped, n_prune = self.drop_minimum(score_drop, layer_mask)

        # Randomly revive n_prune many connections from non-existing connections.
        score_grow = self.grad_dict[layer].abs() * (~layer_mask_dropped) + self._random_uniform(layer_weight.size()) * noise_std
        layer_mask, new_mask = self.grow_maximum(score_grow, layer_mask_dropped, n_prune)

        # update the weight
        if prune.is_pruned(layer):
            prune.remove(layer, 'weight')
        prune.custom_from_mask(layer, 'weight', layer_mask)

        return layer_mask, new_mask

    def reset_momentum(self, masks, new_masks):
        optimizer = self._optimizer
        group = optimizer.param_groups[0]
        momentum = group['momentum']
        if momentum != 0:
            for i, layer in enumerate(self.layers):
                weight_orig = layer.weight_orig
                param_state = optimizer.state[weight_orig]
                new_mask = new_masks[i]
                param_state['momentum_buffer'].mul_(masks[i])
                param_state['momentum_buffer'][new_mask] = self.grad_dict[layer][new_mask] * self._initial_acc_scale


class SparseSnipOptimizer(SparseRigLOptimizer):
    """Implementation of dynamic sparsity optimizers.

    Implementation of Snip
    https://arxiv.org/abs/1810.02340

    Attributes:
        model: torch.nn
        optimizer: torch.Optimizer
        begin_epoch: float, first iteration where masks are updated.
        end_epoch: float, iteration after which no mask is updated.
        train_loader: dataset, contain samples of each batch.
        frequency: int, of mask update operations.
        drop_fraction: float, of connections to drop during each update.
        drop_fraction_anneal: str or None, if supplied used to anneal the drop
            fraction.
        grow_init: str, name of the method used to initialize new connections.
        initial_acc_scale: float, used to scale the gradient when initializing the,
            momentum values of new connections. We hope this will improve training,
            compare to starting from 0 for the new connections. Set this to something
            between 0 and 1 / (1 - momentum). This is because in the current
            implementation of MomentumOptimizer, aggregated values converge to
            1 / (1 - momentum) with constant gradients.
    """

    def __init__(self, model, optimizer, begin_epoch, end_epoch, train_loader, frequency=2, default_sparsity=0.3,
                 drop_fraction=0.5, drop_fraction_anneal='constant',
                 grow_init='zeros', initial_acc_scale=0.):
        super(SparseSnipOptimizer, self).__init__(
            model, optimizer, begin_epoch, end_epoch, train_loader, frequency,
            default_sparsity=default_sparsity,
            drop_fraction=drop_fraction, drop_fraction_anneal=drop_fraction_anneal,
            grow_init=grow_init, initial_acc_scale=initial_acc_scale)

    def update_layer_mask(self, layer, layer_mask, noise_std=1e-5):
        score = self.grad_dict[layer].abs() + noise_std

        # Add noise for slight bit of randomness.
        score_drop = score * layer_mask
        layer_mask_dropped, n_prune = self.drop_minimum(score_drop, layer_mask)

        # Randomly revive n_prune many connections from non-existing connections.
        score_grow = score * (~layer_mask_dropped)
        layer_mask, new_mask = self.grow_maximum(score_grow, layer_mask_dropped, n_prune)

        # update the weight
        if prune.is_pruned(layer):
            prune.remove(layer, 'weight')
        prune.custom_from_mask(layer, 'weight', layer_mask)

        return layer_mask, new_mask


class SparseLpSSOptimizer(SparseRigLOptimizer):
    """Implementation of Lp spherical sparsificaton.

    Attributes:
        model: torch.nn
        optimizer: torch.Optimizer
        begin_epoch: float, first iteration where masks are updated.
        end_epoch: float, iteration after which no mask is updated.
        train_loader: dataset, contain samples of each batch.
        frequency: int, of mask update operations.
        default_sparsity: float, sparsity of initialized ANN
        target_sparsity: float, expected final sparsity
        drop_fraction: float, of connections to drop during each update.
        drop_fraction_anneal: str or None, if supplied used to anneal the drop
              fraction.
        grow_init: str, name of the method used to initialize new connections.
        grow_fraction_d: float, the difference of grow_fraction between sparsity > target_sparsity and
              sparsity < target_sparsity
        initial_acc_scale: float, used to scale the gradient when initializing the,
              momentum values of new connections. We hope this will improve training,
              compare to starting from 0 for the new connections. Set this to something
              between 0 and 1 / (1 - momentum). This is because in the current
              implementation of MomentumOptimizer, aggregated values converge to
              1 / (1 - momentum) with constant gradients.
    """

    def __init__(self, model, optimizer, begin_epoch, end_epoch, train_loader, frequency=50, default_sparsity=0.2, target_sparsity=0.3,
                 drop_fraction=0.1, drop_fraction_anneal='constant', grow_init='zeros', grow_fraction_d=0.05, initial_acc_scale=0.):
        super(SparseLpSSOptimizer, self).__init__(
            model, optimizer, begin_epoch, end_epoch, train_loader, frequency,
            default_sparsity=default_sparsity,
            drop_fraction=drop_fraction, drop_fraction_anneal=drop_fraction_anneal,
            grow_init=grow_init, initial_acc_scale=initial_acc_scale)

        self._grow_init = grow_init
        self._grow_fraction = 0.95
        self._target_sparsity = target_sparsity
        self._grow_fraction_d = grow_fraction_d

    def step(self, criterion=None, inputs=None, labels=None):
        # update parameters
        self._optimizer.step()
        self._optimizer.zero_grad()

        # print('nonzero parameters = {}'.format(co.parameter_statistics(self.model,0)))

        # update fraction
        if self._begin_epoch <= self.global_step / self._batch_epo < self._end_epoch and \
                self.global_step % self._frequency == 0:
            self.compute_gradients(criterion, inputs, labels)  # update the gradient for grow connections
            self.update_topology()
            self._optimizer.zero_grad()
            # update grow fraction
            self.update_grow_fraction()

        self.global_step += 1

    def compute_gradients(self, criterion, inputs, labels):
        """Wraps the compute gradient of passed optimizer."""
        # remove the prune and forward
        for layer in self.layers:
            if prune.is_pruned(layer):
                prune.remove(layer, 'weight')

        # forward
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss = loss.sum()

        # backward and calculate all gradients
        loss.backward()

        # get gradient
        gradient = []
        for layer in self.layers:
            grad, _ = mF.LpNormalize_cnn(layer.weight.grad.data)
            gradient.append(grad)

        self.grad_dict = dict(zip(self.layers, gradient))

    def update_layer_mask(self, layer, layer_mask, noise_std=1e-5):
        layer_weight = layer.weight
        layer_grad = self.grad_dict[layer]

        # Remove weight smaller than adaptive threshold
        layer_mask_dropped, drop_n = sparsify_weight(layer_weight.abs(), layer_mask, self._drop_fraction)

        # Grow weight whose gradient larger than adaptive threshold
        score_grow = layer_grad * (~layer_mask_dropped)
        layer_mask, new_mask = self.grow_maximum(score_grow, layer_mask_dropped, int(drop_n*self._grow_fraction))

        # update the weight
        if prune.is_pruned(layer):
            prune.remove(layer, 'weight')
        prune.custom_from_mask(layer, 'weight', layer_mask)

        return layer_mask, new_mask

    def update_grow_fraction(self):
        # current sparsity
        s = self.mask_sparsity()

        # adjust grow fraction
        if s < self._target_sparsity*(1-0.5*self._drop_fraction):
            self._grow_fraction = (1-self._grow_fraction_d) * s / self._target_sparsity
        else:
            self._grow_fraction = (1+self._grow_fraction_d) * s / self._target_sparsity

    def mask_sparsity(self):
        n_zero = 0
        for mask in self.masks:
            n_zero += (~mask).sum()

        n_para = co.parameter_total(self.model)
        return n_zero/float(n_para)
