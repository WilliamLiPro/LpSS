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
import numpy as np


class VecToTensor(object):
    """Transform the list to tensor"""
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, vec):
        t = torch.tensor(vec, dtype=self.dtype)
        return t


class VecNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` dimensions, this transform
    will normalize each dimension of the input ``torch.*Tensor`` i.e.
    ``input[dim] = (input[dim] - mean[dim]) / std[dim]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor vector of size n to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        return (tensor-self.mean)/self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Crop(object):
    """Randomly select a region from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, padding=4):
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        y = int(np.random.randint(self.padding) - self.padding/2)
        x = int(np.random.randint(self.padding) - self.padding/2)

        img_out = np.zeros((img.size(0), h, w), np.float32)
        y1 = np.clip(y, 0, h)
        y2 = np.clip(y + h, 0, h)
        x1 = np.clip(x, 0, w)
        x2 = np.clip(x + w, 0, w)

        img_out = torch.from_numpy(img_out)
        img_out[:, y1-y: y2-y, x1-x: x2-x] = img[:, y1: y2, x1: x2]

        return img_out


class UniformNoise(object):
    """Randomly zero-centered noise for an image.
    Args:
        amplitude (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, amplitude):
        self.amplitude = amplitude

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        noise = np.random.random((h, w)) * (2 * self.amplitude) - self.amplitude
        noise = torch.from_numpy(noise)
        noise = noise.expand_as(img)
        img = img + noise

        return img