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
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import model.resNet as res
import optimizer.myFunction as myF
import optimizer.transform as transform

from torch.optim.sgd import SGD
from optimizer.lpcgd import LpCGD

from optimizer.sparse_optimizer import SparseLpSSOptimizer, Optimizer
from optimizer.sparse_training import sparse_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Fashion_MNIST_path = 'data'
if __name__ == '__main__':
    """This version slightly improves the accuracy compared with paper, since the transform of image is optimized"""
    EPOCH = 50  # epoch
    BATCH_SIZE = 256  # batch_size
    LR = 0.02

    begin_epoch = EPOCH * 0.2
    end_epoch = EPOCH * 0.8
    frequency = 40

    transform_train = transforms.Compose(
        [transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.13,), (0.31,)),
         transform.Cutout(1, 6)])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.13,), (0.31,))])

    train_set = datasets.FashionMNIST(root=Fashion_MNIST_path, train=True, download=True, transform=transform_train)
    test_set = datasets.FashionMNIST(root=Fashion_MNIST_path, train=False, download=True, transform=transform_test)
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss(reduce=False)
    knots = [0, EPOCH / 6, EPOCH / 3, EPOCH - 3, EPOCH]
    vals1 = [0.04, 2, 0.5, 0.004, 0.0002]
    vals2 = [0.01, 0.4, 0.2, 0.002, 0.0001]

    methods = [SparseLpSSOptimizer]

    names = ['LpSS']

    sparsities = [0.9, 0.8, 0.5]
    lps = [[1.2] * 4 + [1.07] * 18, [1.3] * 4 + [1.15] * 18, [1.4] * 4 + [1.3] * 18]
    for i, sparsity in enumerate(sparsities):
        for j, method in enumerate(methods):
            model = res.wide_resnet18_cifar(input_channels=1, num_classes=10).to(device)
            if j == 0:
                optimizer = LpCGD(model, lps[i], lr=LR, momentum=0.9, nesterov=True)
                sparse = method(model, optimizer, begin_epoch, end_epoch, train_loader, frequency,
                                default_sparsity=0.2, target_sparsity=sparsity,
                                drop_fraction=0.2, drop_fraction_anneal='cosine', grow_fraction_d=0.1)
                vals = vals1
            else:
                optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)
                sparse = method(model, optimizer, begin_epoch, end_epoch, train_loader, frequency,
                                default_sparsity=sparsity,
                                drop_fraction=0.2, drop_fraction_anneal='cosine')
                vals = vals2

            save_path = 'result/Fashion-MNIST ' + names[j] + ' s=' + str(sparsity) + '.pth'
            print('Start training by {}'.format(names[j]))
            summaries, model = sparse_training(model, criterion, sparse, [train_loader, test_loader],
                                    epochs=EPOCH, batch_size=BATCH_SIZE, knots=knots, vals=vals, save_path=save_path)

            epoch_list = []
            lr = []
            train_acc = []
            test_acc = []
            para_num = []
            for summary in summaries:
                epoch_list.append(summary['epoch'])
                lr.append(summary['lr'])
                train_acc.append(summary['train acc'])
                test_acc.append(summary['test acc'])
                para_num.append(summary['para num'])

            relist = [['Fashion-MNIST lp=']] + [lps[i]] + [epoch_list] + [lr] + [train_acc] + [test_acc] + [para_num]
            myF.saveLists(relist, 'result/Fashion-MNIST ' + names[j] + ' s=' + str(sparsity) + '.txt')

    # baseline
    model = res.wide_resnet18_cifar(input_channels=1, num_classes=10).to(device)
    optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0004)
    sparse = Optimizer(model, optimizer, default_sparsity=0)
    save_path = 'result/Fashion-MNIST baseline.pth'
    summaries, model = sparse_training(model, criterion, sparse, [train_loader, test_loader],
                                       epochs=EPOCH, batch_size=BATCH_SIZE, knots=knots, vals=vals2, save_path=save_path)
    epoch_list = []
    lr = []
    train_acc = []
    test_acc = []
    para_num = []
    for summary in summaries:
        epoch_list.append(summary['epoch'])
        lr.append(summary['lr'])
        train_acc.append(summary['train acc'])
        test_acc.append(summary['test acc'])
        para_num.append(summary['para num'])

    relist = [['Fashion-MNIST baseline']] + [epoch_list] + [lr] + [train_acc] + [test_acc] + [para_num]
    myF.saveLists(relist, 'result/Fashion-MNIST baseline.txt')