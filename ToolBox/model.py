import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import Inception3, resnet34, resnet50, vgg11, vgg16
# import DR


class Mag_MNIST(nn.Module):
    # Network architecture defined in '''Towards Evaluating the Robustness of Neural Networks'''
    def __init__(self, in_channels=1, num_classes=10):
        super(Mag_MNIST, self).__init__()
        net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, num_classes)
        )
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


class residual_block_1(nn.Module):
    # 没有维度变化的残差块
    def __init__(self, in_channels, out_channels=None):
        super(residual_block_1, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        self.net = net

    def forward(self, x):
        x1 = self.net(x)
        return F.relu(x1 + x)


class wide_residual_block_1(nn.Module):
    # 没有维度变化的残差块
    def __init__(self, in_channels, out_channels=None):
        super(wide_residual_block_1, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        self.net = net

    def forward(self, x):
        x1 = self.net(x)
        return F.relu(x1 + x)


class residual_block_2(nn.Module):
    # 维度变化的残差块，通过1*1卷积对齐
    def __init__(self, in_channels, out_channels=None):
        super(residual_block_2, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # 以下为project
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        self.net = net

    def forward(self, x):
        x1 = self.net[:-2](x)
        x2 = self.net[-2:](x)
        return F.relu(x1 + x2, inplace=True)


class wide_residual_block_2(nn.Module):
    # 维度变化的残差块，通过1*1卷积对齐
    def __init__(self, in_channels, out_channels=None):
        super(wide_residual_block_2, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # 以下为project
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        self.net = net

    def forward(self, x):
        x1 = self.net[:-2](x)
        x2 = self.net[-2:](x)
        return F.relu(x1 + x2, inplace=True)


class WRN28_10(nn.Module):
    def __init__(self, num_classes=10):
        super(WRN28_10, self).__init__()
        net = nn.Sequential(
            # nn.Conv2d(3, 64, 3, stride=2, padding=3),
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, stride=2, padding=1),
            # layer1
            wide_residual_block_2(16, 160),
            wide_residual_block_1(160),
            wide_residual_block_1(160),
            wide_residual_block_1(160),
            # layer2
            wide_residual_block_2(160, 320),
            wide_residual_block_1(320),
            wide_residual_block_1(320),
            wide_residual_block_1(320),
            # layer3
            wide_residual_block_2(320, 640),
            wide_residual_block_1(640),
            wide_residual_block_1(640),
            wide_residual_block_1(640),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(640, num_classes)
        )
        self.net = net
        self.num_classes = num_classes

    def forward(self, x):
        x = self.net(x)
        return x


class resnet(nn.Module):
    """
    resnet implementation for cifar10 dataset
    n=3 leads to resnet20
    n=5 leads to resnet32
    ...
    """
    def __init__(self, n=3, num_classes=10):
        super(resnet, self).__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        channel = 16
        for i in range(3):
            if i == 0:
                layer = [residual_block_1(16)]
            else:
                layer = [residual_block_2(channel, channel * 2)]
                channel = channel * 2
            for j in range(n - 1):
                layer.append(residual_block_1(channel))
            layer = nn.Sequential(*layer)
            self.net.add_module(f'layer{i+1}', layer)

        self.net.add_module('AdaptivePool', nn.AdaptiveAvgPool2d((1, 1)))
        self.net.add_module('Flatten', nn.Flatten())
        self.net.add_module('Linear', nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.net(x)
        return x


class resnet20(resnet):
    def __init__(self, num_classes=10):
        super(resnet20, self).__init__(n=3, num_classes=num_classes)

class resnet32(resnet):
    def __init__(self, num_classes=10):
        super(resnet32, self).__init__(n=5, num_classes=num_classes)

class resnet56(resnet):
    def __init__(self, num_classes=10):
        super(resnet56, self).__init__(n=9, num_classes=num_classes)


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, w=1):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.w = w
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(self.w*512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(self.w*x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def construct_model(model_name, dataset_name=None, load_state_dict=False, adv=False):
    if dataset_name == 'CIFAR100':
        num_classes = 100
    else:
        num_classes = 10
    if model_name == 'WRN28-10':
        model = WRN28_10(num_classes=num_classes)
    elif model_name == 'magMNIST':
        if dataset_name == 'MNIST':
            model = Mag_MNIST()
        else:
            model = Mag_MNIST(num_classes=43, in_channels=3)
    elif model_name == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif model_name == 'resnet32':
        model = resnet32(num_classes=num_classes)
    # resnet34 and resnet50 are used for high-resolution datasets such as imagenette
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif model_name == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif model_name == 'Inception':
        model = Inception3(num_classes=num_classes)
    elif model_name == 'vgg11':
        model = VGG('VGG11', num_classes=num_classes)
    elif model_name == 'vgg16':
        model = VGG('VGG16', num_classes=num_classes)
    else:
        assert False

    if load_state_dict:
        if dataset_name is None:
            assert False
        if not adv:
            model.load_state_dict(torch.load(f'E:/Checkpoints/normal_models/{model_name}_{dataset_name}.pth'))
        else:
            model.load_state_dict(torch.load(f'E:/Checkpoints/adv_models/{model_name}_{dataset_name}.pth'))

    return model


class SequenceModel(nn.Module):
    def __init__(self, modules:list):
        super(SequenceModel, self).__init__()
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def train(self, mode: bool = True):
        for module in self.modules:
            module.train()
        self.training = True

    def eval(self):
        for module in self.modules:
            module.eval()
        self.training = False
            
    def __getitem__(self, item):
        return self.modules[item]


class Ensemble(nn.Module):
    def __init__(self, endpoints):
        super(Ensemble, self).__init__()
        self.endpoints = endpoints

    def forward(self, x):
        outputs = []
        for endpoint in self.endpoints:
            outputs.append(endpoint(x))
        output = torch.stack(outputs)
        output = torch.sum(output, dim=0)
        return output

    def train(self, mode: bool = True):
        for endpoint in self.endpoints:
            endpoint.train()

    def eval(self):
        for endpoint in self.endpoints:
            endpoint.eval()






if __name__ == '__main__':
    construct_model('vgg11')


