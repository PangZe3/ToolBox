import random
import math
import torch
from torchvision.models import vgg11, vgg16, resnet50, resnet34, resnet101, wide_resnet50_2, inception_v3
from torch import nn
from torchvision import transforms
# from model import Mask_LeNet, Mask_Mag_MNIST
import numpy as np


def int_shape(x):
    return list(map(int, x.shape))


def show_tensor_as_pil(tensor):
    from torchvision import transforms
    tensor = tensor.clone().cpu()
    inv_transform = transforms.Compose([
        # transforms.Normalize((-1,), (2,)),
        transforms.ToPILImage()
    ])
    ts = int_shape(tensor)
    if len(ts) == 4:
        tensor = tensor.permute(1, 2, 0, 3)
        ts = int_shape(tensor)
        tensor = torch.reshape(tensor, (*ts[:2], ts[2] * ts[3]))
    tensor_pil = inv_transform(tensor)
    tensor_pil.show()


def save_tensor_as_pil(tensor, name):
    from torchvision import transforms
    tensor = tensor.clone().cpu()
    inv_transform = transforms.Compose([
        # transforms.Normalize((-1,), (2,)),
        transforms.ToPILImage()
    ])
    ts = int_shape(tensor)
    if len(ts) == 4:
        tensor = tensor.permute(1, 2, 0, 3)
        ts = int_shape(tensor)
        tensor = torch.reshape(tensor, (*ts[:2], ts[2] * ts[3]))
    tensor_pil = inv_transform(tensor)
    tensor_pil.save(name)


def get_time_stamp():
    import time
    return time.strftime('%m%d%H%M%S', time.localtime())



