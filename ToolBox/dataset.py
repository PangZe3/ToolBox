from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, GTSRB, CIFAR100, ImageFolder, SVHN, CelebA
from torchvision import transforms
import torch
from torch.utils.data import Subset


def get_dataset(dataset_name, train=True, transform=None):

    if dataset_name == 'MNIST':
        if transform is None:
            transform = transforms.ToTensor()
        ori_dataset = MNIST(root='E:/Datasets',
                            train=train,
                            download=True,
                            transform=transform)
    elif dataset_name == 'FashionMNIST':
        if transform is None:
            transform = transforms.ToTensor()
        ori_dataset = FashionMNIST(root='E:/Datasets',
                                   train=train,
                                   download=True,
                                   transform=transform)
    elif dataset_name == 'CIFAR10':
        if transform is None:
            transform = transforms.ToTensor()
        ori_dataset = CIFAR10(root='E:/Datasets',
                              train=train,
                              download=True,
                              transform=transform)
    elif dataset_name == 'CIFAR100':
        if transform is None:
            transform = transforms.ToTensor()
        ori_dataset = CIFAR100(root='E:/Datasets',
                               train=train,
                               download=True,
                               transform=transform)
    elif dataset_name == 'GTSRB':
        if transform is None:
            transform = transforms.ToTensor()
        ori_dataset = GTSRB(root='E:/Datasets',
                            split='train' if train else 'test',
                            download=True,
                            transform=transform)
    elif dataset_name == 'imagenette':
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(160),
                transforms.ToTensor(),
            ])
        if train:
            ori_dataset = ImageFolder(root='E:/Datasets/imagenette2-160/train',
                                      transform=transform)
        else:
            ori_dataset = ImageFolder(root='E:/Datasets/imagenette2-160/val',
                                      transform=transform)
    elif dataset_name == 'SVHN':
        if transform is None:
            transform = transforms.ToTensor()
        ori_dataset = SVHN(root='E:/Datasets',
                           split='train' if train else 'test',
                           download=True,
                           transform=transform)
    elif dataset_name == 'CelebA':
        if transform is None:
            transform = transforms.ToTensor()
        # CelbA dataset has a split for validation set
        ori_dataset = CelebA(root='E:/Datasets',
                             split='train' if train else 'test',
                             target_type='identity',
                             download=True,
                             transform=transform)

    else:
        assert False

    return ori_dataset


def get_subset(dataset, capacity):
    # indices = torch.randint(0, len(dataset), (capacity,))
    indices = torch.randperm(len(dataset))[:capacity]
    sub_set = Subset(dataset, indices)
    return sub_set


if __name__ == '__main__':
    train_set = get_dataset('CelebA')
    a = 1