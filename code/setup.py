import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import *
from schedules import *
from torch.optim import lr_scheduler
import pandas as pd
from tqdm import tqdm
import argparse

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=8)
cifar_dataloader = {'train':trainloader, 'val':testloader}

#MNIST Dataset
mnist_dataloader = {
    'train': torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=64,
        shuffle=True,
        num_workers=8
    ),
    'val': torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=64,
        shuffle=False,
        num_workers=8
    )
}

dataset_loader = {'mnist':mnist_dataloader, 'cifar10':cifar_dataloader}

class Experiment:
    def __init__(
        self,
        experiment_id = 1,
        model = None,
        epochs = 300,
        use_gpu = False,
    ):
        self.experimentID = experiment_id
        self.model = model
        self.epochs = epochs
        self.useGPU = torch.cuda.is_available() and use_gpu
        self.error = {'epoch':[-1], 'train':[100], 'val': [100]}
        self.loss = {'epoch':[-1], 'train':[100], 'val': [100]}
        self.lr = {'epoch':[-1], 'lr':[-1]}
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.learningRateSchedule = self.getLearningRateSchedule(self.optimizer)
        self.dataset = None
        if self.useGPU:
            model = nn.DataParallel(model)
            self.model = model.cuda()
        else:
            self.model = model

    def getLearningRateSchedule(self, optimizer):
        if self.experimentID == 0:
            return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[100, 175, 225],
            gamma=0.1
        )
        elif self.experimentID == 1:
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=75,
                gamma=0.1
            )
        elif self.experimentID == 2:
            return CosineWithRestartLR(
                self.optimizer,
                min_lr=1e-4,
                max_lr=0.1,
                restart_interval=50,
                restart_multiplier=1,
                amplitude_decay=1
            )
        elif self.experimentID == 3:
            return CosineWithRestartLR(
                self.optimizer,
                min_lr=1e-4,
                max_lr=0.1,
                restart_interval=10,
                restart_multiplier=2,
                amplitude_decay=1
            )
        elif self.experimentID == 4:
            return CosineWithRestartLR(
                self.optimizer,
                min_lr=1e-4,
                max_lr=0.1,
                restart_interval=50,
                restart_multiplier=1,
                amplitude_decay=0.5
            )
        elif self.experimentID == 5:
            return CosineWithRestartLR(
                self.optimizer,
                min_lr=1e-4,
                max_lr=0.1,
                restart_interval=10,
                restart_multiplier=2,
                amplitude_decay=0.5
            )
        elif self.experimentID == 6:
            return AdaptiveLR(
                self.optimizer,
                start_lr = 0.1,
                mu=0.99,
                eps=0.001
            )
        elif self.experimentID == 7:
            return AdaptiveLR(
                self.optimizer,
                start_lr=0.1,
                mu=0.9,
                eps=0.001
            )
        elif self.experimentID == 8:
            return AdaptiveLR(
                self.optimizer,
                start_lr=0.1,
                mu=0.99,
                eps=0.001,
                train=False
            )
        elif self.experimentID == 9:
            return AdaptiveLR(
                self.optimizer,
                start_lr=0.1,
                mu=0.9,
                eps=0.001,
                train=False
            )
        elif self.experimentID == 10:
            return AdaptiveLR(
                self.optimizer,
                start_lr=0.1,
                mu=0.95,
                eps=0.001,
                train=True
            )
        elif self.experimentID == 11:
            return AdaptiveLR(
                self.optimizer,
                start_lr=0.1,
                mu=0.95,
                eps=0.001,
                train=False
            )
        else:
            raise Exception('experiment not ready')



    def train_cifar(self):
        for epoch in tqdm(range(self.epochs)):
            self.error['epoch'].append(epoch)
            self.loss['epoch'].append(epoch)
            self.lr['epoch'].append(epoch)
            for phase in tqdm(['train', 'val']):
                running_loss = 0.0
                running_error = 0.0

                if phase == 'train':
                    if isinstance(self.learningRateSchedule, AdaptiveLR):
                        if epoch == 0:
                            self.learningRateSchedule.step()
                        else:
                            if self.learningRateSchedule.train:
                                self.learningRateSchedule.step(
                                    self.error['train'][epoch],
                                    self.error['train'][epoch - 1]
                                )
                            else:
                                self.learningRateSchedule.step(
                                    self.error['val'][epoch],
                                    self.error['val'][epoch - 1]
                                )
                    else:
                        self.learningRateSchedule.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                for i, (inputs, labels) in \
                        tqdm(enumerate(cifar_dataloader[phase], 0)):
                    if self.useGPU:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    running_loss += loss.data[0]
                    running_error += torch.sum(preds != labels.data)

                self.error[phase].append(
                    running_error / len(cifar_dataloader[phase])
                )
                self.loss[phase].append(
                    running_loss / len(cifar_dataloader[phase])
                )
            self.lr['lr'].append(self.learningRateSchedule.get_lr()[0])
            print "\n\n\nepoch: ", epoch, " lr: ",\
                self.lr['lr'][epoch+1], " train error: ",\
                self.error['train'][epoch+1], " val error: ",\
                self.error['val'][epoch+1], " train loss: ", \
                self.loss['train'][epoch+1], " val loss: ", \
                self.loss['val'][epoch+1]
        self.generateStatistics()

    def train_mnist(self):
        pass


    def train(self, dataset='cifar10'):
        self.dataset = dataset
        if dataset == 'cifar10':
            self.train_cifar()
        elif dataset == 'mnist':
            self.train_mnist()
        else:
            pass

    def generateStatistics(self):
        error = {'epoch':self.error['epoch'][1:], 'train':self.error['train'][1:],\
                'val':self.error['val'][1:]}

        loss = {'epoch':self.loss['epoch'][1:], 'train':self.loss['train'][1:],\
                'val':self.loss['val'][1:]}

        lr = {'epoch':self.lr['epoch'][1:], 'lr':self.lr['lr'][1:]}

        error_df = pd.DataFrame(error)
        loss_df = pd.DataFrame(loss)
        lr_df = pd.DataFrame(lr)
        error_df.to_csv(
            'statistics/experiment' + str(self.experimentID) + '_error' \
                + '_' + self.dataset +'.csv',
            index=False
        )
        loss_df.to_csv(
            'statistics/experiment' + str(self.experimentID) + '_loss' \
                + '_' + self.dataset +'.csv',
            index=False,
        )
        lr_df.to_csv(
            'statistics/experiment' + str(self.experimentID) + '_lr' \
                + '_' + self.dataset +'.csv',
            index=False,
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ALeRTS'
    )
    parser.add_argument(
        '--exp',
        default=0,
        type=int,
        help='which exp to run',
        dest='exp_id'
    )
    obj = parser.parse_args()
    exp = Experiment(
        experiment_id = obj.exp_id,
        model = VGG('VGG16'),
        epochs = 250,
        use_gpu = True,
    )
    exp.train('cifar10')
