from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import numpy as np
import matplotlib 
from utils import blackbox_allmodels, select_valsubset, test_acc_allmodels
from preact_resnet import PreActResNet18
import resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
model_names.append('resnet18')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--adadelta_lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
parser.add_argument('--adam_lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate for adam optimizer (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--val_seed', type=int, default=2, metavar='S',
                        help='random seed for data splitting(default: 2)')
parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
parser.add_argument('--save_name', type=str, default='',
                        help='Name for Saving the current Model')
parser.add_argument('--save_dir', type=str, default='output_dir',
                        help='Directory for saving the model')
parser.add_argument('--checkpt', type=str, default='_best',
                        help='which checkpoint to load')
parser.add_argument('--models_num', type=int, default=3,
                        help='number of models to train jointly (default: 3)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
						
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'batch_size': args.batch_size}
if use_cuda:
    kwargs.update({'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True},
                 )
                 
def test(model, device, test_loader, split_type):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(split_type+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)
    return acc 
    
dataset1 = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor()
    ]), download=True)

main_train_set, val_set = torch.utils.data.random_split(dataset1, [len(dataset1)-5000, 5000],torch.Generator().manual_seed(args.val_seed))
train_loader = torch.utils.data.DataLoader(main_train_set,**kwargs)
val_loader = torch.utils.data.DataLoader(val_set,**kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=args.batch_size, shuffle=False, pin_memory=True)

models_list = []
for i in range(0,args.models_num):    
    if args.arch == 'resnet18':
        model1 = PreActResNet18()
    else:
        model1 = resnet.__dict__[args.arch]()
    #model1 = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model1.cuda()
    model1.load_state_dict(torch.load(args.save_dir+'/model'+str(i)+args.checkpt+'.pt'))
    model1.eval()
    models_list.append(model1)
    
acc_test_all = 0 
acc_val_all = 0 
for i in range(0,args.models_num):
    print('********* Model '+str(i)+' *********')
    acc = test(models_list[i], device, test_loader,'test')
    acc_test_all += acc 
    acc = test(models_list[i], device, val_loader,'val')
    acc_val_all += acc 

print('********* Average *********')

acc_test_all = acc_test_all / args.models_num
acc_val_all = acc_val_all / args.models_num

print('Test set average acc: {:.4f}\n'.format(acc_test_all))
print('Val set average acc: {:.4f}\n'.format(acc_val_all))