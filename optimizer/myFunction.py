import os
import torch
import torch.nn.functional as F
import numpy as np

from collections import namedtuple

import time
import matplotlib.pyplot as plt

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LpNormalize_cnn(input, p=2, cp=1, eps=1e-6):
    r'''Calculate the unit vector on Lp sphere
    :param input:  tensor of weight, dims should be >= 2
    :param p: the Lp parameter of weight
    :param cp: the p power of current input, that means input = c*w^cp
    :param eps:
    :return: output = input/norm_d, norm_d = norm(input, p/cp)
    '''
    dim = input.dim()

    norm_d = LpNorm_cnn(input, p, cp, eps)
    inv_norm_d = 1 / norm_d

    if dim == 2:
        output = input.mul(inv_norm_d.view(input.size(0), 1))
    elif dim == 3:
        output = input.mul(inv_norm_d.view(input.size(0), 1, 1))
    elif dim == 4:
        output = input.mul(inv_norm_d.view(input.size(0), 1, 1, 1))
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return output, norm_d


def LpNorm_cnn(input, p=2, cp=1, eps=1e-6):
    r'''Calculate the Lp norm of weights
        :param input:  tensor of weight, dims should be >= 2, and the dim 0 is channels
        :param p: the Lp parameter of weight
        :param cp: the p power of current input, that means input = c*w^cp
        :param eps:
        :return: output = input/norm_d, norm_d = norm(input, p/cp)
        '''
    dim = input.dim()

    if dim == 2:
        norm_d = input.abs().pow(p / cp).sum(1).pow(cp / p).add(eps)
    elif dim == 3:
        norm_d = input.abs().pow(p / cp).sum(2).sum(1).pow(cp / p).add(eps)
    elif dim == 4:
        norm_d = input.abs().pow(p / cp).sum(3).sum(2).sum(1).pow(cp / p).add(eps)
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return norm_d


def LpNormalize_layer(input, p=2, cp=1, eps=1e-6):
    r'''Calculate the unit vector on Lp sphere (layer as a single vector)
    :param input:  tensor of weight, dims should be >= 2
    :param p: the Lp parameter of weight
    :param cp: the p power of current input, that means input = c*w^cp
    :param eps:
    :return: output = input/norm_d, norm_d = norm(input, p/cp)
    '''
    dim = input.dim()

    if dim >= 2 and dim <= 4:
        norm_d = input.abs().pow(p/cp).sum().pow(cp/p).add(eps)
        inv_norm_d = 1/norm_d
        output = input.mul(inv_norm_d)
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return output, inv_norm_d


def Hoyer_layer_sparsity(input, eps=1e-8):
    # Hoyer’s sparsity of a layer's weight
    # the average sparsity of weight in a layer
    dim = input.dim()
    abs_in = input.abs()
    d = np.prod(input.size()[1:])
    sqrt_d = np.sqrt(d)

    if dim == 2:
        output = abs_in.sum(1).div(abs_in.pow(2).sum(1).pow(0.5).add(eps)).sub(sqrt_d).div(1-sqrt_d).mean(0)
    elif dim == 3:
        output = abs_in.sum(2).sum(1).div(abs_in.pow(2).sum(2).sum(1).pow(0.5).add(eps)).sub(sqrt_d).div(1-sqrt_d).mean(0)
    elif dim == 4:
        output = abs_in.sum(3).sum(2).sum(1).div(abs_in.pow(2).sum(3).sum(2).sum(1).pow(0.5).add(eps)).sub(sqrt_d).div(1-sqrt_d).mean(0)
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return output


def Hoyer_layervec_sparsity(input, eps=1e-8):
    # Hoyer’s sparsity of a layer's weight
    # the average sparsity of weight in a layer
    dim = input.dim()
    abs_in = input.abs()
    d = np.prod(input.size())
    sqrt_d = np.sqrt(d)

    if dim >= 2 and dim <= 4:
        output = abs_in.div(abs_in.pow(2).pow(0.5).add(eps)).sub(sqrt_d).div(1-sqrt_d)
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return output


def Hoyer_net_sparsity(model):
    # Hoyer’s sparsity of whole network
    # the average sparsity of weight in whole network
    weight_list = get_weight(model)
    w_sparsity = []
    num_w = []  # number of weight in each layer

    for name, weight in weight_list:
        if weight.dim() < 2:
            continue

        # sparsity
        c_sparse = Hoyer_layer_sparsity(weight.data).item()
        w_sparsity.append(c_sparse)
        num_w.append(np.prod(weight.data.size()))

    return np.average(w_sparsity, weights=num_w)


def Hoyer_net_ll_sparsity(model):
    # Hoyer’s sparsity of whole network
    # the average sparsity of weight in whole network
    weight_list = get_weight(model)
    w_sparsity = []
    num_w = []  # number of weight in each layer

    for name, weight in weight_list:
        if weight.dim() < 2:
            continue

        # sparsity
        c_sparse = Hoyer_layervec_sparsity(weight.data).item()
        w_sparsity.append(c_sparse)
        num_w.append(np.prod(weight.data.size()))

    return np.average(w_sparsity, weights=num_w)


def Hoyer_activation_sparsity(input):
    # Hoyer’s sparsity of a layer's activation
    # the average sparsity of activation in a layer
    return Hoyer_layer_sparsity(input)


def sparsify_weight(w, mask, h=0.1, eps=1e-8):
    '''Weight sparsification by setting the small element to zero
    Args:
        w (torch.tensor): the weight for sparsification
        mask (torch.tensor): the mask of no activated weight
        h (torch.tensor/float, optional): the weight for sparsification (default: 0.1)

    :return: w (sparse), mask
        '''
    wa = w.abs()
    nmask_f = (~mask).float()
    dim = wa.dim()
    if dim == 2:
        hh = wa.mul(nmask_f).sum(1).div(nmask_f.sum(1).add(eps)).mul(h)
        mask = wa < hh.view(wa.size(0), 1)
        w.masked_fill_(mask, 0)
    elif dim == 3:
        hh = wa.mul(nmask_f).sum(2).sum(1).div(nmask_f.sum(2).sum(1).add(eps)).mul(h)
        mask = wa < hh.view(wa.size(0), 1, 1)
        w.masked_fill_(mask, 0)
    elif dim == 4:
        hh = wa.mul(nmask_f).sum(3).sum(2).sum(1).div(nmask_f.sum(3).sum(2).sum(1).add(eps)).mul(h)
        mask = wa < hh.view(wa.size(0), 1, 1, 1)
        w.masked_fill_(mask, 0)
    else:
        raise ValueError('Expected dimension of input 2 <= dims <=4, got {}'.format(dim))

    return w, mask


def sparsify_weight_ll(w, h=0.1):
    '''Weight sparsification by setting the small element to zero
    Args:
        w (torch.tensor): the weight for sparsification
        h (torch.tensor/float, optional): the weight for sparsification (default: 0.1)

    :return: mask
        '''
    wa = w.abs()
    ws = wa.mul(wa)
    dim = ws.dim()
    if dim >= 2 and dim <= 4:
        hh = ws.mean().sqrt().mul(h)
        mask = wa < hh
        w.masked_fill_(mask, 0)
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return w, mask


def sparsify_grad(g, mask, h=0.1, eps=1e-10):
    '''grow connection by activate large gradient
    Args:
        g (torch.tensor): the gradient of weight
        mask (torch.tensor): the mask of no activated weight
        h (torch.tensor/float, optional): the weight for sparsification (default: 0.1)

    :return: mask
        '''
    ga = g.abs()
    nmask_f = (~mask).float()
    dim = ga.dim()
    if dim == 2:
        hh = ga.mul(nmask_f).sum(1).div(nmask_f.sum(1).add(eps)).mul(h)
        mask = (ga < hh.view(ga.size(0), 1)) & mask
    elif dim == 3:
        hh = ga.mul(nmask_f).sum(2).sum(1).div(nmask_f.sum(2).sum(1).add(eps)).mul(h)
        mask = (ga < hh.view(ga.size(0), 1, 1)) & mask
    elif dim == 4:
        hh = ga.mul(nmask_f).sum(3).sum(2).sum(1).div(nmask_f.sum(3).sum(2).sum(1).add(eps)).mul(h)
        mask = (ga < hh.view(ga.size(0), 1, 1, 1)) & mask
    else:
        raise ValueError('Expected dimension of input 2 <= dims <=4, got {}'.format(dim))

    return mask


def sparsify_grad_ll(g, mask, h=0.1, eps=1e-8):
    '''grow connection by activate large gradient
    Args:
        g (torch.tensor): the gradient of weight
        mask (torch.tensor): the mask of no activated weight
        h (torch.tensor/float, optional): the weight for sparsification (default: 0.1)

    :return: output = input(input.abs()<h*input.abs().mean())=0
        '''
    ga = g.abs()
    mask_f = mask.float()
    nmask_f = 1 - mask_f
    dim = ga.dim()
    if dim >= 2 and dim <= 4:
        hh = ga.mul(nmask_f).sum().div(nmask_f.sum().add(eps)).mul(h)
        mask = (ga < hh) & mask
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return mask


def orthogonalProjection(w, x):
    # the projection of x orthogonal to the w
    size_w = w.size()
    size_x = x.size()

    if size_w != size_x:
        raise ValueError('Expected size of x should be same as w {}, got {}'.format(size_w, size_x))

    dim = w.dim()

    if dim == 2:
        r = w.mul(x).sum(1)
        p = x.sub(w.mul(r.view(size_w[0], 1)))
    elif dim == 3:
        r = w.mul(x).sum(2).sum(1)
    elif dim == 4:
        r = w.mul(x).sum(3).sum(2).sum(1)
    else:
        raise ValueError('Expected input 2 <= dims <=4, got {}'.format(dim))

    return r


def get_weight(model):
    '''
    获得模型的权重列表
    :param model:
    :return:
    '''
    weight_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            weight_list.append(weight)
    return weight_list


def saveLists(lists, textname):
    # save the lists to certain text
    file = open(textname,'w')
    for data in lists:
        m = len(data)
        for i, p in enumerate(data):
            file.write(str(p))
            if i<m-1:
                file.write(',')
        file.write('\n')
    file.close()