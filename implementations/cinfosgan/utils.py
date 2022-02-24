import random
import time
import datetime
import sys
from typing import Tuple, cast

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

def convolution_output_shape(dim: int, kernel_size: int, padding: int, stride: int, dilation: int):
    return (dim + 2 * padding - (dilation * (kernel_size - 1) - 1) - 1) // stride + 1

def lower_bound_convolution_input_shape(dim: int, kernel_size: int, padding: int, stride: int, dilation: int):
    return (dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

def output_shape_of_conv2d(layer: torch.nn.Conv2d, input_shape: tuple) -> Tuple[int,int]:
    assert len(input_shape) == 2
    height = input_shape[0]
    width = input_shape[1]
    padding: Tuple[int,...] = cast(Tuple[int,...], layer.padding)
    return (
        convolution_output_shape(height, layer.kernel_size[0], padding[0], layer.stride[0], layer.dilation[0]),
        convolution_output_shape(width, layer.kernel_size[1], padding[1], layer.stride[1], layer.dilation[1])
    )

def lb_input_shape_of_conv2d(layer: torch.nn.Conv2d, output_shape: tuple) -> Tuple[int,int]:
    assert len(output_shape) == 2
    height = output_shape[0]
    width = output_shape[1]
    padding: Tuple[int,...] = cast(Tuple[int,...], layer.padding)
    return (
        lower_bound_convolution_input_shape(height, layer.kernel_size[0], padding[0], layer.stride[0], layer.dilation[0]),
        lower_bound_convolution_input_shape(width, layer.kernel_size[1], padding[1], layer.stride[1], layer.dilation[1])
    )

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
