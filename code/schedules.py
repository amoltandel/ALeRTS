import numpy as np
import pandas as pd
import torch
import torch.optim as optim

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

class CosineWithRestartLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        min_lr=1e-5,
        max_lr=0.01,
        restart_interval=50,
        restart_multiplier=1,
        amplitude_decay=1,
        last_epoch=-1
    ):
        self.minLR = min_lr
        self.maxLR = max_lr
        self.restartInterval = restart_interval
        self.restartMultiplier = restart_multiplier
        self.amplitudeDecay = amplitude_decay
        self.nextRestartEpoch = self.restartInterval
        self.lastRestartEpoch = 0
        self.currentAmplitude = self.maxLR
        self.last_epoch = last_epoch
        super(CosineWithRestartLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [self.lambda_fun()
                for base_lr in self.base_lrs]

    def lambda_fun(self):
        if self.last_epoch == 0:
            return self.maxLR
        # update
        if self.last_epoch == self.nextRestartEpoch:
            self.nextRestartEpoch = (self.last_epoch - self.lastRestartEpoch) * self.restartMultiplier + self.last_epoch
            self.lastRestartEpoch = self.last_epoch
            self.currentAmplitude *= self.amplitudeDecay
            return self.currentAmplitude

        lr = self.minLR +\
            0.5 * (self.currentAmplitude - self.minLR) *\
             (1 + np.cos((self.last_epoch - self.lastRestartEpoch) / float(
                self.nextRestartEpoch - self.lastRestartEpoch) * np.pi)
            )
        return lr

class AdaptiveLR():
    def __init__(
        self,
        optimizer,
        start_lr = 0.1,
        mu=0.99,
        eps=0.1,
        last_epoch=-1,
        train=True
    ):
        self.optimizer = optimizer
        self.eps = eps
        self.mu = mu
        self.last_epoch = last_epoch
        self.previousLR = start_lr
        self.currentError = 1
        self.previousError = 1
        self.train = train
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def step(self, err_i_1=1, err_i_2=1, epoch=None):
        self.currentError = err_i_1
        self.previousError = err_i_2
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [self.adaptiveRate()
                for base_lr in self.base_lrs]

    def adaptiveRate(self):
        if self.last_epoch == 0:
            return self.previousLR
        self.previousLR = abs(self.mu * self.previousLR - self.eps * (self.currentError - self.previousError))
        return self.previousLR


class VanillaUpdateLR(AdaptiveLR):
    def __init__(
        self,
        optimizer,
        start_lr,
        eps,
        last_epoch=-1,
        train=True
    ):
        super(VanillaUpdateLR).__init__(
            optimizer,
            start_lr,
            1.0,
            eps,
            last_epoch,
            train
        )

class MomentumUpdateLR(AdaptiveLR):
    def __init__(
        self,
        optimizer,
        start_lr = 0.1,
        mu=0.99,
        eps=0.1,
        last_epoch=-1,
        train=True
    ):
        self.errcache = 0
        super(MomentumUpdateLR).__init__(
            optimizer,
            start_lr,
            mu,
            eps,
            last_epoch,
            train
        )

    def adaptiveRate(self):
        if self.last_epoch == 0:
            return self.previousLR
        self.errcache = self.mu * self.errcache + self.eps * (self.currentError - self.previousError)
        self.previousLR = abs(self.previousLR - self.errcache)
        return self.previousLR

class AdagradLR(AdaptiveLR):
    def __init__(
        self,
        optimizer,
        start_lr = 0.1,
        mu=0.99,
        eps=0.1,
        last_epoch=-1,
        train=True
    ):
        self.errcache = 0
        super(AdagradLR).__init__(
            optimizer,
            start_lr,
            1.0,
            eps,
            last_epoch,
            train
        )

    def adaptiveRate(self):
        if self.last_epoch == 0:
            return self.previousLR

        self.errcache += (self.currentError - self.previousError) ** 2
        self.previousLR = abs(self.previousLR - self.eps * (self.currentError - self.previousError) / np.sqrt(self.errcache + 1e-10))
        return self.previousLR

class RMSpropLR(AdaptiveLR):
    def __init__(
        self,
        optimizer,
        start_lr = 0.1,
        mu=0.99,
        eps=0.1,
        last_epoch=-1,
        train=True
    ):
        self.errcache = 0
        super(RMSpropLR).__init__(
            optimizer,
            start_lr,
            mu,
            eps,
            last_epoch,
            train
        )

    def adaptiveRate(self):
        if self.last_epoch == 0:
            return self.previousLR

        self.errcache = self.mu * self.errcache + \
            (1-self.mu) * (self.currentError - self.previousError) ** 2
        self.previousLR = abs(self.previousLR - self.eps * (self.currentError - self.previousError) / np.sqrt(self.errcache + 1e-10))
        return self.previousLR

class AdamLR(AdaptiveLR):
    def __init__(
        self,
        optimizer,
        start_lr = 0.1,
        mu1=0.99,
        mu2=0.99,
        eps=0.1,
        last_epoch=-1,
        train=True
    ):
        self.errcache = 0
        self.errcache2 = 0
        self.mu2 = mu2
        super(AdamLR).__init__(
            optimizer,
            start_lr,
            mu,
            eps,
            last_epoch,
            train
        )

    def adaptiveRate(self):
        if self.last_epoch == 0:
            return self.previousLR
        derr = self.currentError - self.previousError
        self.errcache = self.mu * self.errcache + \
            (1-self.mu) * derr
        self.errcache2 = self.mu * self.errcache + \
            (1-self.mu) * (derr ** 2)
        self.previousLR = abs(self.previousLR - self.eps * (self.errcache) / np.sqrt(self.errcache2 + 1e-10))
        return self.previousLR

def load_experiment(exp_id=0):
    err_file = "../statistics/experiment"+ str(exp_id) + "_error_cifar10.csv"
    loss_file = "../statistics/experiment"+ str(exp_id) + "_loss_cifar10.csv"
    lr_file = "../statistics/experiment"+ str(exp_id) + "_lr_cifar10.csv"

    lr_data = pd.read_csv(lr_file)
    epochs = lr_data['epoch'].values.tolist()
    lrs = lr_data['lr'].values.tolist()

    lr_data = pd.read_csv(err_file)
    train_err = lr_data['train'].values.tolist()
    val_err = lr_data['val'].values.tolist()

    lr_data = pd.read_csv(loss_file)
    train_loss = lr_data['train'].values.tolist()
    val_loss = lr_data['val'].values.tolist()

    return epochs, lrs, train_err, val_err, train_loss, val_loss

def plot_lr_data(epochs, lrs):
    import matplotlib.pyplot as plt
    plt.axis(([min(epochs), max(epochs) + 1, min(lrs), max(lrs) * 10]))
    plt.semilogy(epochs, lrs, color='r', label='Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.show()

def plot_err_data(epochs, train, val):
    import matplotlib.pyplot as plt
    plt.axis(([min(epochs), max(epochs) + 1, min(min(train), min(val)), max(max(train), max(val))]))
    plt.plot(epochs, train, color='r', label='Training error')
    plt.plot(epochs, val, color='b', label='Validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def plot_loss_data(epochs, train, val):
    import matplotlib.pyplot as plt
    plt.axis(([min(epochs), max(epochs) + 1, min(min(train), min(val)), max(max(train), max(val))]))
    plt.semilogy(epochs, train, color='r', label='Train loss')
    plt.semilogy(epochs, val, color='b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross entropy Loss')
    plt.legend()
    plt.show()

def plot_data(exp_id):
    epochs, lrs, train_err, val_err, train_loss, val_loss = load_experiment(exp_id)
    plot_lr_data(epochs, lrs)
    plot_err_data(epochs, train_err, val_err)
    plot_loss_data(epochs, train_loss, val_loss)

if __name__ == '__main__':
    plot_data(8)
    exit(1)
    from torchvision.models import AlexNet
    model = AlexNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineWithRestartLR(
                optimizer,
                min_lr=1e-4,
                max_lr=0.1,
                restart_interval=10,
                restart_multiplier=2,
                amplitude_decay=1
     )
    # scheduler = AdaptiveLR(
    #     optimizer,
    #     start_lr = 0.01,
    #     mu=0.99,
    #     eps=0.1,
    #     last_epoch=-1
    # )
    errs = [1, 0.359375, 0.328125, 0.28125, 0.21875, 0.234375, 0.1875, 0.140625, 0.140625, 0.109375, 0.125, 0.140625, 0.140625, 0.125, 0.140625, 0.171875, 0.15625, 0.09375, 0.140625, 0.140625, 0.15625, 0.078125, 0.09375, 0.09375, 0.0625, 0.15625, 0.078125, 0.09375, 0.1875, 0.078125, 0.09375, 0.09375, 0.125, 0.125, 0.125, 0.09375, 0.140625, 0.109375, 0.09375, 0.078125, 0.125, 0.09375, 0.078125, 0.09375, 0.078125, 0.171875, 0.109375, 0.078125, 0.140625, 0.078125, 0.125, 0.078125, 0.0625, 0.109375, 0.109375, 0.09375, 0.171875, 0.046875, 0.078125, 0.0625, 0.03125, 0.09375, 0.15625, 0.109375, 0.109375, 0.078125, 0.0625, 0.046875, 0.09375, 0.03125, 0.078125, 0.0625, 0.078125, 0.140625, 0.046875, 0.015625, 0.09375, 0.109375, 0.0625, 0.09375, 0.0625, 0.046875, 0.046875, 0.078125, 0.046875, 0.03125, 0.078125, 0.078125, 0.03125, 0.078125, 0.078125, 0.140625, 0.109375, 0.0625, 0.078125, 0.03125, 0.09375, 0.046875, 0.046875, 0.078125, 0.09375, 0.046875, 0.0625, 0.078125, 0.0625, 0.0625, 0.015625, 0.015625, 0.109375, 0.078125, 0.109375, 0.078125, 0.140625, 0.046875, 0.03125, 0.046875, 0.15625, 0.125, 0.046875, 0.046875, 0.09375, 0.09375, 0.015625, 0.109375, 0.078125, 0.078125, 0.09375, 0.078125, 0.0625, 0.046875, 0.109375, 0.15625, 0.046875, 0.0625, 0.046875, 0.078125, 0.03125, 0.0625, 0.0625, 0.09375, 0.125, 0.03125, 0.078125, 0.078125, 0.046875, 0.0625, 0.078125, 0.078125, 0.046875, 0.125, 0.046875, 0.0625, 0, 0.015625, 0.015625, 0.015625, 0, 0.015625, 0, 0.015625, 0.015625, 0, 0.015625, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0.015625, 0, 0, 0.015625, 0.03125, 0, 0, 0, 0, 0, 0, 0, 0.03125, 0, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0.015625, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0, 0, 0.015625, 0.015625, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.015625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    epochs = []
    lrs = []
    for i in range(200):
        if isinstance(scheduler, AdaptiveLR):
            scheduler.step(errs[i+1], errs[i])
        elif isinstance(scheduler, CosineWithRestartLR):
            scheduler.step()
        else:
            raise Exception('cannot identify the scheduler')
        epochs.append(i)
        lrs.append(scheduler.get_lr()[0])
        print i, scheduler.last_epoch, scheduler.get_lr()[0], optimizer.param_groups[0]['lr']

    plot_lr_data(epochs, lrs)
