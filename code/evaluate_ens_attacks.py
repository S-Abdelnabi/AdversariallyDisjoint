from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import resnet
from torch.autograd import Variable
import os
import numpy as np
from PIL import Image
import matplotlib 
from utils import blackbox_allmodels, select_valsubset, test_acc_allmodels
from preact_resnet import PreActResNet18
import numpy as np
import ensemble_models
from itertools import combinations 
import sys 
from statistics import mean 

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
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
parser.add_argument('--adadelta_lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
parser.add_argument('--adam_lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate for adam optimizer (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--val_seed', type=int, default=2, metavar='S',
                        help='random seed for data splitting(default: 2)')
parser.add_argument('--gradient_val_seed', type=int, default=3, metavar='S',
                        help='random seed for selecting validation set for gradients transferrability checking(default: 3)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
parser.add_argument('--save_name', type=str, default='',
                        help='Name for Saving the current Model')
parser.add_argument('--save_dir', type=str, default='output_dir',
                        help='Directory for saving the model')
parser.add_argument('--checkpt', type=str, default='_best',
                        help='which checkpoint to load')
                        
parser.add_argument('--model_type', type=str, default='conv',
                        help='Conv or FC model')
parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer to use')
parser.add_argument('--gradient_weight', type=float, default=1,
                        help='default weight for the gradient loss')
parser.add_argument('--models_num', type=int, default=3,
                        help='number of models to train jointly (default: 3)')
parser.add_argument('--epsilon', type=float, default=0.15,
                        help='float value of the adv attack')
parser.add_argument('--attack_type', type=str, default='fgm',
                        help='fgm or fgsm')
parser.add_argument('--alpha', type=float, default=0.003,
                        help='float value of alpha in bim adv attack')
parser.add_argument('--alpha_rfgsm', type=float, default=0.003,
                        help='float value of alpha in r fgsm adv attack')
parser.add_argument('--decay', type=float, default=1.0,
                        help='float value of decay in MI-FGSM')
parser.add_argument('--bim_itrs', type=int, default=0,
                        help='number of iterations in the BIM attack')
parser.add_argument('--mi_itrs', type=int, default=10,
                        help='number of iterations in the BIM attack')
parser.add_argument('--if_print', type=int, default=0,
                        help='whether to print individual values')
parser.add_argument('--instance_per_class_val', type=int, default=300, metavar='S',
                        help='instances per class in the gradient val set, default: 100')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
#### C&W attack #####
parser.add_argument('--cw_c', type=float, default=0.01,
                        help='float value of c in c&w attack')
parser.add_argument('--cw_conf', type=float, default=0,
                        help='float value of confidence in c&w attack')
parser.add_argument('--cw_lr', type=float, default=1e-2,
                        help='float value of confidence in c&w attack')
parser.add_argument('--cw_iters', type=int, default=1000,
                        help='int value of the max number of iterations in c&w attack')  

#### EAD #####                        
parser.add_argument('--ead_c', type=float, default=20,
                        help='float value of c in EAD')
parser.add_argument('--ead_beta', type=float, default=0.01,
                        help='float value of beta in EAD')
parser.add_argument('--ead_conf', type=float, default=0,
                        help='float value of confidence in EAD')
parser.add_argument('--ead_lr', type=float, default=0.01,
                        help='float value of confidence in EAD')
parser.add_argument('--ead_iters', type=int, default=1000,
                        help='int value of the max number of iterations in EAD')
parser.add_argument('--ead_rule', type=str, default='EN',
                        help='decision rule for EAD, en or L1')                        
						
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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(split_type+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)
    return acc 
    
def fgsm_attack(model_src, target, image, epsilon, idx_list=None):
    output_src = model_src(image) if idx_list==None else model_src.forward_subset(image, idx_list)
       
    loss = F.cross_entropy(output_src, target)
    model_src.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def ll_fgsm_attack(model_src, image, epsilon, idx_list=None):
    output_src = model_src(image) if idx_list==None else model_src.forward_subset(image, idx_list)
    _, labels = torch.min(output_src.data, 1)
    labels = labels.detach_()
    
    loss = F.cross_entropy(output_src, labels)
    model_src.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgm_attack(model_src, target, image, epsilon, idx_list = None):
    output_src =model_src(image) if idx_list==None else model_src.forward_subset(image, idx_list)
       
    loss = F.cross_entropy(output_src, target)
    model_src.zero_grad()
    loss.backward()
    data_grad = image.grad.data
        
    data_grad_norm = torch.norm(data_grad,dim=(2,3)).view(data_grad.size(0),data_grad.size(1),1,1)
    data_grad_normalized = data_grad/data_grad_norm 

    perturbed_image = image + epsilon*data_grad_normalized.view_as(image)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image



def mi_fgsm(model, images, labels, eps, decay, iters=0, alpha=0, idx_list=None) :
    images = images.to(device)
    labels = labels.to(device)
    momentum = torch.zeros_like(images).to(device)
    clamp_max = 1.0
    
    if iters == 0 :
        iters = int(min(eps*256 + 4, 1.25*eps*256))
    if alpha==0:
        alpha = eps/iters

    orig_images = images
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images) if idx_list==None else model.forward_subset(images, idx_list)

        model.zero_grad()
        cost = F.cross_entropy(outputs, labels)
        cost.backward()
        
        grad = images.grad.data
        grad_norm = torch.norm(grad, p=1)
        grad /= grad_norm
        grad += momentum*decay
        momentum = grad
            
        attack_images = images + alpha*grad.sign()
        
        a = torch.clamp(orig_images - eps, min=0)
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        c = (b > orig_images+eps).float()*(orig_images+eps) + (orig_images+eps >= b).float()*b
        images = torch.clamp(c, max=clamp_max).detach_()
            
    return images
    


def r_fgsm_attack(model, labels, images, eps, alpha, idx_list=None) :

    images = images.to(device)
    labels = labels.to(device)
    
    random_noise = alpha*torch.randn(size=images.size()).sign().cuda()   
    images_new = torch.clamp(images.detach() + random_noise,0,1)
    images_new.requires_grad = True
    
    outputs = model(images_new) if idx_list==None else model.forward_subset(images_new, idx_list)
    
    model.zero_grad()
    cost = F.cross_entropy(outputs, labels)#.to(device)
    cost.backward()
    
    attack_images = images_new + (eps-alpha)*images_new.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

def pgd_random(model, images, labels, eps, alpha, iters=0,idx_list=None) :
    images = images.to(device)
    labels = labels.to(device)
    clamp_max = 1.0
    
    if iters == 0 :
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(eps*256 + 4, 1.25*eps*256))
        
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1)

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images) if idx_list==None else model.forward_subset(adv_images, idx_list)
        model.zero_grad()
        cost = F.cross_entropy(outputs, labels).to(device)
        cost.backward()
        grad = adv_images.grad.data 

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images

#CW attack 
def cw_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01, idx_list=None) :

    # Define f-function
    def f(x) :
        outputs = model(x) if idx_list==None else model.forward_subset(x, idx_list)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels > 0)
        
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %% ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

#ead attack 
def ead_attack(model, images, labels, beta, decision_rule, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01, idx_list=None) :
    def ISTA(new, old):
        with torch.no_grad():
            diff = new - old
            var_beta = torch.FloatTensor(np.ones(shape=diff.shape, dtype=float) * beta).to(device)
            # test if the perturbation is out of bound. If it is, then reduce the perturbation by beta
            cropped_diff = torch.max(torch.abs(diff) - var_beta, torch.zeros(diff.shape, device=device)) * diff.sign().to(device)
            fist_new = old + cropped_diff
        return torch.clamp(input=fist_new, min=0.0, max=1.0)
        
    def attack_achieved(pre_softmax, target_class):
        if targeted :
            pre_softmax[target_class] -= kappa
        else: 
            pre_softmax[target_class] += kappa
        max_class = torch.argmax(pre_softmax)
        if targeted :
            return torch.equal(max_class, target_class)
        else: 
            return not torch.equal(max_class, target_class)
    # Define f-function
    def f(x) :
        outputs = model(x) if idx_list==None else model.forward_subset(x, idx_list)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels > 0)
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa), outputs
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa), outputs
    
    slack = images.detach()
    slack.requires_grad = True 
    # optimize the slack variable
    optimizer = optim.SGD([slack], lr=learning_rate)
    old_image = slack.clone()    
    ##initialize 
    best_elastic = [1e10] * args.batch_size
    best_perturbation = torch.zeros(images.size()).cuda()
    flag = np.zeros((images.size(0),))

    for step in range(max_iter) :

        loss1 = nn.MSELoss(reduction='sum')(slack, images)
        loss2 = torch.sum(c*f(slack)[0])

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        new_image = ISTA(slack, images)
        slack.data = new_image.data + ((step / (step + 3.0)) * (new_image - old_image)).data
        old_image = new_image.clone()


        # calculate the loss for decision
        l1dist = torch.sum(torch.abs(new_image - images), [1, 2, 3])
        l2dist = torch.sum((new_image - images) ** 2, [1, 2, 3])
        #target_loss = torch.max((output - 1e10 * targets_one_hot).max(1)[0] - (output * targets_one_hot).sum(1), -1 * kappa_t)
        target_loss, outputs = f(new_image)

        if decision_rule=='EN':
            decision_loss = beta * l1dist + l2dist + c * target_loss
        else:
            decision_loss = beta * l1dist + c * target_loss

        # Update best results
        for i, (dist, score, img) in enumerate(zip(decision_loss.detach(), outputs.detach(), new_image.detach())):
            success = attack_achieved(score, labels[i])
            #print(success)
            if dist < best_elastic[i] and success:               
                best_elastic[i] = dist
                best_perturbation[i] = img
                flag[i] = 1
                        
        print('- Learning Progress : %2.2f %% ' %((step+1)/max_iter*100), end='\r')
    for i in range(0,images.size(0)):
        if flag[i] == 0:
            #print('Attack did not succeed')
            best_perturbation[i] = new_image[i].detach()

    return best_perturbation
    
def test_attack( ens_model, models, device, test_loader, epsilon, split_type, idx_list=None):
    # Accuracy counter
    correct = np.zeros((len(models),))
    count_imgs = 0

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        #print(torch.max(data))
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Cal Attacks
        if args.attack_type == 'fgsm':        
            perturbed_data = fgsm_attack(ens_model, target,data, epsilon,idx_list=idx_list)
        elif args.attack_type == 'rfgsm':        
            perturbed_data = r_fgsm_attack(ens_model, target,data, epsilon, epsilon/2,idx_list=idx_list)
        elif args.attack_type == 'llfgsm':        
            perturbed_data = ll_fgsm_attack(ens_model,data, epsilon,idx_list=idx_list)
        elif args.attack_type == 'fgm':
            perturbed_data = fgm_attack(ens_model, target, data, epsilon,idx_list=idx_list)
       elif args.attack_type == 'mifgsm':
            perturbed_data = mi_fgsm(ens_model, data, target, args.epsilon, args.decay, args.mi_itrs,args.alpha,idx_list=idx_list)
        elif args.attack_type == 'pgd':
            perturbed_data = pgd_random(ens_model, data, target, args.epsilon, args.alpha, args.bim_itrs,idx_list=idx_list)
        elif args.attack_type == 'cw':
            perturbed_data = cw_l2_attack(ens_model, data, target, targeted=False, c=args.cw_c, kappa=args.cw_conf, max_iter=args.cw_iters, learning_rate=args.cw_lr,idx_list=idx_list)            
        elif args.attack_type == 'ead':        
            perturbed_data = ead_attack(ens_model, data, target, args.ead_beta, args.ead_rule, targeted=False, c=args.ead_c, kappa=args.ead_conf, max_iter=args.ead_iters, learning_rate=args.ead_lr,idx_list=idx_list)
        else:
            print('Attack is not implemented')            
        # Re-classify the perturbed image
        count_imgs += data.size(0)
        for model_idx in range(0,len(models)):
            output_target = models[model_idx](perturbed_data)
            output_target = F.log_softmax(output_target, dim=1)
            final_pred = output_target.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct[model_idx] += final_pred.eq(target.view_as(final_pred)).sum().item()
        
    # Calculate final accuracy for this epsilon
    final_acc = correct/count_imgs 

    return final_acc

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

max_combination = len(models_list)
print(max_combination)
ens_model = ensemble_models.MyEnsemble(models_list)

    
##### load dataset ######
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


acc_test = np.zeros((args.models_num,args.models_num)) 
acc_val = np.zeros((args.models_num,args.models_num))
acc_test= test_attack(ens_model, models_list, device, test_loader, args.epsilon, 'Test')*100 

orig_stdout = sys.stdout
f = open('ens_out_' + str(len(models_list))+'models.txt', 'w')
sys.stdout = f


print('*** Test ***')

print('* all models *')
print(acc_test)
print('test avg: '+str(np.mean(acc_test)))
print('test std: '+str(np.std(acc_test)))
print('test min: '+str(np.min(acc_test)))
f.flush()

list_i = [i for i in range(0,len(models_list))]
for i in range(2, max_combination):
    print('-------------------------')
    print('** ' + str(i) + ' combinations **')
    comb = list(combinations(list_i, i))
    average_list = []
    for one_comb in comb: 
        acc_test= test_attack(ens_model, models_list, device, test_loader, args.epsilon, 'Test',idx_list=one_comb)*100 
        print('* src models: ' + ','.join(str(e) for e in one_comb) + ' *')
        print(acc_test)
        print('test avg: '+str(np.mean(acc_test)))
        print('test std: '+str(np.std(acc_test)))
        print('test min: '+str(np.min(acc_test)))
        average_list.append(np.mean(acc_test))
        f.flush()
    print('Average of average for ' + str(i) + ' combinations: '+str(mean(average_list)))

sys.stdout = orig_stdout
f.close()