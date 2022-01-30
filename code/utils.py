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
from torch import randperm, default_generator
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image.detach()

def fgm_attack(image, epsilon, data_grad):
    data_grad_norm = torch.norm(data_grad,dim=(2,3)).view(data_grad.size(0),data_grad.size(1),1,1)
    data_grad_normalized = data_grad/data_grad_norm 

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*data_grad_normalized.view_as(image)
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image.detach()

def pgd_random(device, model, images, labels, eps, alpha, iters=0):
    clamp_max = 1.0
    if iters == 0 :
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(eps*256 + 4, 1.25*eps*256))
        
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1)

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        model.zero_grad()
        cost = F.cross_entropy(outputs, labels).to(device)
        cost.backward()
        grad = adv_images.grad.data 
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def fast_rand_fgsm(model_src, target, image, epsilon, alpha):
    delta = torch.zeros_like(image).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta.data = clamp(delta, 0-image, 1-image)
    delta.requires_grad = True
    output_src = model_src(image+delta)
       
    loss = F.cross_entropy(output_src, target)
    model_src.zero_grad()
    loss.backward()
    grad = delta.grad.detach()
    delta.data = clamp(delta + alpha * torch.sign(grad), torch.tensor([-epsilon]).cuda(), torch.tensor([epsilon]).cuda())
    delta.data = clamp(delta, 0-image, 1-image)
    delta = delta.detach()

    perturbed_image = image + delta
    return perturbed_image

def fast_rand_fgsm_allmodels(models_list, target, image, epsilon, alpha):
    perturbed_data_list = []
    for model in models_list:
        perturbed_data = fast_rand_fgsm(models_list[i], target, image, epsilon, alpha)
        perturbed_data_list.append(perturbed_data)
    return perturbed_data_list
    
def test_attack(models, src_index, device, test_loader, epsilon, attack_type):
    correct = np.zeros((len(models),))   

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output_src = models[src_index](data)

        loss = F.cross_entropy(output_src, target)
        models[src_index].zero_grad()
        loss.backward()
        data_grad = data.grad.detach()
        if attack_type == 'fgsm':        
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack_type == 'fgm':
            perturbed_data = fgm_attack(data, epsilon, data_grad)
        with torch.no_grad():
            for model_idx in range(0,len(models)):
                output_target = models[model_idx](perturbed_data)
                output_target = F.log_softmax(output_target, dim=1)
                final_pred = output_target.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct[model_idx] += final_pred.eq(target.view_as(final_pred)).sum().item()
    final_acc = 100. * correct/float(len(test_loader.dataset))
    return final_acc

def test_attack_whitebox(model, device, test_loader, epsilon, attack_type):
    correct = 0  

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output_src = model(data)

        loss = F.cross_entropy(output_src, target)
        models[src_index].zero_grad()
        loss.backward()
        data_grad = data.grad.detach()
        if attack_type == 'fgsm':        
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack_type == 'fgm':
            perturbed_data = fgm_attack(data, epsilon, data_grad)
        with torch.no_grad():
            output_target = model(perturbed_data)
            output_target = F.log_softmax(output_target, dim=1)
            final_pred = output_target.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()
    final_acc = 100. * correct/float(len(test_loader.dataset))
    return final_acc

def test_attack_iterative( models, src_index, device, test_loader, epsilon, alpha, iters):
    correct = np.zeros((len(models),))
    # Accuracy counter 

    # Loop over all examples in test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        perturbed_data = pgd_random(device, models[src_index], data, target, epsilon, alpha, iters)
        with torch.no_grad():
            for model_idx in range(0,len(models)):
                output_target = models[model_idx](perturbed_data)
                output_target = F.log_softmax(output_target, dim=1)
                final_pred = output_target.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct[model_idx] += final_pred.eq(target.view_as(final_pred)).sum().item()
    final_acc = 100. * correct/float(len(test_loader.dataset))
    return final_acc

def test_attack_iterative_whitebox( model, device, test_loader, epsilon, alpha, iters):
    correct = 0
    # Accuracy counter 

    # Loop over all examples in test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        perturbed_data = pgd_random(device, model, data, target, epsilon, alpha, iters)
        with torch.no_grad():
            output_target = model(perturbed_data)
            output_target = F.log_softmax(output_target, dim=1)
            final_pred = output_target.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()
    final_acc = 100. * correct/float(len(test_loader.dataset))
    return final_acc
    
def blackbox_allmodels(models_list,device,val_loader,epsilon,attack_type,alpha=0.025,iters=7):
    acc_val = np.zeros((len(models_list),len(models_list)))
    for i in range(0,len(models_list)):
        if attack_type == 'pgd':
            acc_val[i,:] = test_attack_iterative(models_list,i, device, val_loader,epsilon,alpha, iters)
        else:
            acc_val[i,:] = test_attack(models_list,i, device, val_loader,epsilon,attack_type)
    acc_val_transferrable = acc_val - np.multiply(acc_val,np.identity(len(models_list)))
    avg_per_model = np.sum(acc_val_transferrable,axis=1)/(len(models_list)-1)
    return np.mean(avg_per_model)

def blackbox_external_model(models_list, external_model, device, val_loader,epsilon,attack_type):
    acc_val = np.zeros((len(models_list),))
    for i in range(0,len(models_list)):
        acc_val[i] = test_attack( external_model,models_list[i], device, val_loader,epsilon,attack_type)
    return np.mean(acc_val)

def whitebox_allmodels(models_list,device,val_loader,epsilon,attack_type,alpha=0.025,iters=7):
    acc_val = np.zeros((len(models_list),))
    for i in range(0,len(models_list)):
        if attack_type == 'pgd':
            acc_val[i] = test_attack_iterative_whitebox(models_list[i], device, val_loader,epsilon,alpha, iters)
        else:
            acc_val[i] = test_attack_whitebox(models_list[i], device, val_loader,epsilon,attack_type)
    return np.mean(acc_val)

def test_acc(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    return acc 

def test_acc_allmodels(models_list, device, test_loader):
    acc_models = np.zeros((len(models_list),))
    for i in range(0,len(models_list)):   
        acc_models[i] = test_acc( models_list[i], device, test_loader)
    acc_all_models = np.sum(acc_models)/len(models_list)
    return acc_all_models    

def select_valsubset(dataset,instances_per_class,generator=default_generator):
    for i in range(0,10):
        idx = torch.nonzero(dataset.targets==i)
        perm_idx = randperm(len(idx), generator=generator).tolist()
        if i==0:
            val_sub = Subset(dataset,perm_idx[0:instances_per_class])
            train_sub = Subset(dataset,perm_idx[instances_per_class:len(perm_idx)])
        else:
            val_sub = ConcatDataset([val_sub,Subset(dataset,perm_idx[0:instances_per_class])])
            train_sub = ConcatDataset([train_sub, Subset(dataset,perm_idx[instances_per_class:len(perm_idx)])])
    return train_sub, val_sub   
