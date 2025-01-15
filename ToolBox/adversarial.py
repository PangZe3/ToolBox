import torchattacks
from tqdm import tqdm
import torch


def test_robustness(attack: str, model, dataloader, eps=0.03, targeted=False, surrogate_model=None,
                    **kwargs):
    """
    test robustness of model
    :param model: target classifier (or sequence of defense processing and classifier)
    :param dataloader: dataloader of test data
    :param eps: magnitude of attack, CW attack has no designated epsilon
    :param targeted: whether implement targeted attack
    :param surrogate_model: if designated, use surrogate model to generate adversarial examples
    """
    adv_num = len(dataloader.dataset)
    if surrogate_model is None:
        surrogate_model = model
    atk = instantiate_attack(surrogate_model, attack, eps, targeted, **kwargs)

    accuracy = 0
    for i, (img, labels) in enumerate(dataloader):
        print(f'\rattack {attack}, generating batch [{i + 1}/{adv_num // dataloader.batch_size + 1}]', end=' ')
        img = img.cuda()
        labels = labels.cuda()
        if attack != 'none':
            atk.set_device('cuda:0')
            adv_img = atk(img, labels)
            logits = model(adv_img)
        else:
            logits = model(img)
        prediction = torch.argmax(logits, dim=1)
        accuracy += torch.sum(prediction == labels)
    print('\r ', end= '')
    accuracy = accuracy / adv_num
    return accuracy


def instantiate_attack(model, attack:str, eps=0.03, targeted=False, **kwargs):
    model.eval()
    model.cuda()
    atk = None
    if attack == 'FGSM':
        atk = torchattacks.FGSM(model, eps)
    elif attack == 'PGD':
        if 'alpha' not in kwargs.keys():
            alpha = 2 / 255
        else:
            alpha = kwargs['alpha']
        if 'steps' not in kwargs.keys():
            steps = 30
        else:
            steps = kwargs['steps']
        # print(f'default steps for PGD is 30')
        atk = torchattacks.PGD(model, eps, alpha=alpha, steps=steps)
    elif attack == 'CW':
        c = 1 if 'c' not in kwargs.keys() else kwargs['c']
        kappa = 0 if 'kappa' not in kwargs.keys() else kwargs['kappa']
        steps = 100 if 'steps' not in kwargs.keys() else kwargs['steps']
        lr = 0.01 if 'lr' not in kwargs.keys() else kwargs['lr']
        # print(f'default steps for CW is 100')
        atk = torchattacks.CW(model, c, kappa, steps, lr)
    elif attack == 'AutoAttack':
        norm = 'Linf' if 'Linf' not in kwargs.keys() else kwargs['norm']
        n_classes = 10 if 'n_classes' not in kwargs.keys() else kwargs['n_classes']
        atk = torchattacks.AutoAttack(model, norm=norm, n_classes=n_classes)
    elif attack == 'none':
        pass
    else:
        print('not implemented')
        assert False

    if targeted and attack != 'none':
        if attack in ['AutoAttack']:
            print(f'{attack} has no targeted mode')
        else:
            atk.set_mode_targeted_random(quiet=True)

    return atk









