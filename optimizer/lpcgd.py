import torch
from torch.optim.optimizer import Optimizer, required
import myFunction as myF

class LpCGD(Optimizer):
    r"""Implements Lp constrained gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        net_lp (list): Lp norm of weight for each layer
        lr (float): learning rate
        lr_decay (float): decay of learning rate (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, model, net_lp, lp_layers: list = ['conv', 'fc'], min_input_channel: int = 2, forbid_layers: int = 1,
                 lr=required, lr_decay: float = 0,
                 momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov=False):

        # other set
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_decay < 0.0:
            raise ValueError("Invalid lr_decay rate: {}".format(lr_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lr_decay=lr_decay, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.sum_iter = 0
        self.general_norm = 1
        self.net_lp = net_lp
        self.lp_layers = lp_layers
        self.min_input_channel = min_input_channel
        self.forbid_layers = forbid_layers * 2

        namelist = []
        for name, param in model.named_parameters():
            namelist.append(name)

        self.name_list = namelist

        super(LpCGD, self).__init__(model.parameters(), defaults)

        self.weightNormalize()

    def weightNormalize(self):
        # normalized weight

        # fill the lp for normalization
        layer_n = len(self.name_list)
        net_lp = [None] * layer_n

        count = 0
        for i, name in enumerate(self.name_list):
            if i >= layer_n - self.forbid_layers:
                break

            if (True in [lp_layer in name for lp_layer in self.lp_layers]) and 'weight' in name:
                net_lp[i] = self.net_lp[count]
                count += 1

        self.net_lp = net_lp

        # normalized weight
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.data is None or net_lp[i] is None:
                    continue
                cp = p.data
                if cp.dim() < 2 or cp.dim() > 4 or cp.size(1) < self.min_input_channel:
                    net_lp[i] = None
                    continue
                p.data, _ = myF.LpNormalize_cnn(cp, net_lp[i])

    def __setstate__(self, state):
        super(LpCGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        net_lp = self.net_lp

        for group in self.param_groups:
            # update learning ratio
            cur_lr = group['lr'] / (1 + group['lr_decay'] * self.sum_iter)
            self.sum_iter += 1

            # params
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                # gradient
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # normalized gradient
                lp = net_lp[i]
                if lp is not None:
                    # this is the update of normalized weight
                    # update feature vector
                    w = p.data

                    if 'feature_vec' not in param_state:
                        ftv = param_state['feature_vec'] = w.sign().mul(w.abs().pow(lp-1))
                    else:
                        ftv = param_state['feature_vec']

                    d_pc, inv_norm_d = myF.LpNormalize_cnn(d_p, lp, lp-1)
                    ftv = ftv.mul(1-cur_lr).add(-cur_lr, d_pc)

                    ftv, _ = myF.LpNormalize_cnn(ftv, lp, lp - 1)
                    param_state['feature_vec'] = ftv
                    p.data = ftv.sign().mul(ftv.abs().pow(1/(lp - 1)))
                else:
                    # this is bias and other parameters
                    b = p.data
                    b.add_(-cur_lr, d_p)

        return loss


class LpCGDw(Optimizer):
    r"""Implements Lp constrained gradient descent (optionally with momentum).

    This version directly update the weight instead of updating the feature vector defined as:
    v(w)=w^(p-1)

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        net_lp (list): Lp norm of weight for each layer
        lr (float): learning rate
        lr_decay (float): decay of learning rate (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, model, net_lp, lp_layers: list = ['conv', 'fc'], forbid_layers: int = 1,
                 lr=required, lr_decay=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        # other set
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_decay < 0.0:
            raise ValueError("Invalid lr_decay rate: {}".format(lr_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lr_decay=lr_decay, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # get the namelist of parameters
        self.sum_iter = 0
        self.general_norm = 1
        self.net_lp = net_lp
        self.lp_layers = lp_layers
        self.forbid_layers = forbid_layers

        namelist = []
        for name, param in model.named_parameters():
            namelist.append(name)

        self.name_list = namelist

        super(LpCGDw, self).__init__(model.parameters(), defaults)

        self.weightNormalize()

    def weightNormalize(self):
        # normalized weight

        # fill the lp for normalization
        layer_n = len(self.name_list)
        net_lp = [None] * layer_n

        count = 0
        for i, name in enumerate(self.name_list):
            if i >= layer_n - self.forbid_layers:
                break

            if (True in [lp_layer in name for lp_layer in self.lp_layers]) and 'weight' in name:
                net_lp[i] = self.net_lp[count]
                count += 1

        self.net_lp = net_lp

        # normalized weight
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.data is None or net_lp[i] is None:
                    continue
                cp = p.data
                if cp.dim() < 2 or cp.dim() > 4:
                    net_lp[i] = None
                    continue
                p.data, _ = myF.LpNormalize_cnn(cp, net_lp[i])

    def __setstate__(self, state):
        super(LpCGDw, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        net_lp = self.net_lp

        for group in self.param_groups:
            # update learning ratio
            cur_lr = group['lr'] / (1 + group['lr_decay'] * self.sum_iter)
            self.sum_iter += 1

            # params
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                # gradient
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # normalized gradient
                lp = net_lp[i]
                if lp is not None:
                    # this is weight update of convolution layer and full connected layer
                    # normalized weight
                    w = p.data

                    # unit gradient on p/(p-1) norm space
                    d_pc, inv_norm_d = myF.LpNormalize_cnn(d_p, lp, lp - 1)

                    # w(t+1) = (1 - lr ) * w(t) - lr * gradient ^ (1/(p-1))
                    d_pc = d_pc.sign().mul(d_pc.abs().pow(1 / (lp - 1)))

                    # update the weight
                    w = w.mul(1 - cur_lr).add(-cur_lr, d_pc)
                    p.data, _ = myF.LpNormalize_cnn(w, lp, 1)
                else:
                    # this is bias update or other kind of weight
                    b = p.data
                    b.add_(-cur_lr, d_p)

        return loss