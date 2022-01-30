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
from utils import blackbox_allmodels, select_valsubset, test_acc_allmodels
#from scipy.misc import comb
from scipy.special import perm
from preact_resnet import PreActResNet18

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
model_names.append('resnet18')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
parser.add_argument('--adadelta_lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--val_seed', type=int, default=2, metavar='S',
                        help='random seed for data splitting(default: 2)')
  
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')                      
parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
parser.add_argument('--save_name', type=str, default='',
                        help='Name for Saving the current Model')
parser.add_argument('--save_dir', type=str, default='output_dir',
                        help='Directory for saving the model')
parser.add_argument('--model_type', type=str, default='conv',
                        help='Conv or FC model')
parser.add_argument('--leaky_relu_slope', type=float, default=0.1,
                        help='slope of the negative part of the leaky relu')
                        
parser.add_argument('--optimizer_type', type=str, default='adam',
                        help='optimizer to use')  
parser.add_argument('--lr_sched', type=str, default='decay',
                        help='normal decay or cyclic scheduler')
### Adam params 
parser.add_argument('--adam_lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate for adam optimizer (default: 0.0001)')
### Adam decay scheduler params 
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--step_size', type=int, default=1,
                        help='Learning rate step size (default: 1)')
### Adam cyclic scheduler params 
parser.add_argument('--adam_max_lr', type=float, default=0.0006, metavar='LR',
                        help='max learning rate for adam optimizer (default: 0.0006)')
   
parser.add_argument('--models_num', type=int, default=3,
                        help='number of models to train jointly (default: 3)')

###SGD parames     
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_max', default=0.2, type=float,
                    metavar='LR', help='max learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
                        
#### which losses to use ####
parser.add_argument('--include_gradient_angle', type=int, default=0,
                        help='whether to add the cosine in loss')
parser.add_argument('--include_transfer_loss', type=int, default=1,
                        help='whether to use the transfer loss')
parser.add_argument('--include_transfer_sgn_loss', type=int, default=1,
                        help='whether to use the transfer loss based on the gradient sign')
#train against fixed model                        
parser.add_argument('--include_fixed_model', type=int, default=0,
                        help='whether to train against a fixed external model')
parser.add_argument('--fixed_model_path', type=str, default='model_external1',
                        help='name of the pretrained external model to load')
                        
#### gradient angle loss ####                        
parser.add_argument('--start_gradient_loss', type=int, default='0',
                        help='number of epochs after which we start the gradient loss')
parser.add_argument('--end_gradient_loss', type=int, default='20',
                        help='number of epochs after which we stop training with the gradient angle loss')
parser.add_argument('--gradient_loss', type=str, default='dot',
                        help='whether to use pairwise or averaged loss')
parser.add_argument('--gradient_threshold', type=float, default=-1,
                        help='the required cos value between models gradients to reach')
#### soft sign functions ####                    
parser.add_argument('--soft_sgn', type=str, default='tanh',
                        help='the approximation used for the sign')
                        
#### gradient transfer loss ####
parser.add_argument('--start_transfer_loss', type=int, default='0',
                        help='number of epochs after which we start the gradient transfer loss')
parser.add_argument('--start_transfer_sgn_loss', type=int, default='0',
                        help='number of epochs after which we start the gradient transfer loss for the soft sign')
                        
#### weights of losses ####
parser.add_argument('--gradient_weight', type=float, default=1,
                        help='default weight for the gradient loss')
parser.add_argument('--gradient_weight_fixed', type=float, default=3,
                        help='default weight for the gradient loss of the fixed model')
parser.add_argument('--classify_weight', type=float, default=1,
                        help='default weight for the classification loss')
parser.add_argument('--transfer_loss_weight', type=float, default=1,
                        help='default weight for the transfer loss')
parser.add_argument('--transfer_loss_weight_fixed', type=float, default=1,
                        help='default weight for the transfer loss')
parser.add_argument('--transfer_loss_sgn_weight', type=float, default=1,
                        help='default weight for the transfer loss of soft sign')
parser.add_argument('--transfer_loss_sgn_weight_fixed', type=float, default=1,
                        help='default weight for the transfer loss of soft sign')

#### check blackbox accuracy interval ####
parser.add_argument('--tranfser_check_interval', type=int, default='5',
                        help='cycle to check the current transferrability rate of attacks across models')
parser.add_argument('--fgm_epsilon', type=float, default=1.5,
                        help='float value of the adv attack - FGM')  
parser.add_argument('--fgsm_epsilon', type=float, default=0.03,
                        help='float value of the adv attack - FGSM')
parser.add_argument('--fgsm_epsilon_training', type=float, default=0.03,
                        help='float value of the adv attack during training of the transfer loss - FGSM') 
parser.add_argument('--fgm_epsilon_training', type=float, default=2,
                        help='float value of the adv attack during training of the transfer loss - FGM')                     
parser.add_argument('--attack_type_check', type=str, default='single_step',
                        help='the attack to check against during training')  

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)') 
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')                    

                        
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if use_cuda else "cpu")
tanh = nn.Tanh()

kwargs = {'batch_size': args.batch_size}
if use_cuda:
    kwargs.update({'num_workers': args.workers,
                   'pin_memory': True,
                   'shuffle': True},
                 )
                  
if not os.path.isdir(args.save_dir): 
    try:
        os.mkdir(args.save_dir)
    except OSError:
        print ("Creation of the directory %s failed" % args.save_dir)
    else:
        print ("Successfully created the directory %s " % args.save_dir)

def shifted_relu_loss(loss, threshold):
    relative_loss = loss - threshold
    relative_loss_clamped = torch.clamp(relative_loss,min=0)
    return relative_loss_clamped

def normalize_one_gradient(data_grad1):
    data_grad1_norm = torch.norm(data_grad1,dim=(2,3)).view(data_grad1.size(0),data_grad1.size(1),1,1)
    data_grad1_normalized = data_grad1/data_grad1_norm 
    if torch.sum(torch.isnan(data_grad1_normalized))> 0:
        print('nan in norm')
        return data_grad1 
    return data_grad1_normalized
    
def cosine_loss(data_grads_list, bsz): 
    data_grads_norm_list = []
    mean_dot_list = []
    mean_dot_avg = 0
    for i in range(0,args.models_num):
        data_grad_normalized = normalize_one_gradient(data_grads_list[i])
        data_grads_norm_list.append(data_grad_normalized)
    count = 0
    for i in range(0,args.models_num):
        for j in range(i+1,args.models_num):
            if i==j:
                continue
            count = count + 1
            dot_product = torch.dot(torch.flatten(data_grads_norm_list[i]), torch.flatten(data_grads_norm_list[j]))
            dot_product = dot_product / bsz
            dot_product = dot_product / 3
            mean_dot_list.append(dot_product.data)
            mean_dot_avg = mean_dot_avg + dot_product
    mean_dot_avg = mean_dot_avg / count    
    return mean_dot_list,mean_dot_avg

def compute_input_sgn_grads(data_grads_list):
    data_grads_sgn_list = []
    for i in range(0, args.models_num):
        #soft_grad_sign = tanh(data_grads_list[i]*10000000)
        #soft_grad_sign = tanh(data_grads_list[i]*10000)
        soft_grad_sign = tanh(data_grads_list[i]*20000)
        #print(soft_grad_sign*args.fgsm_epsilon_training)
        data_grads_sgn_list.append(soft_grad_sign)
    return data_grads_sgn_list

def compute_loss_increase(model,data,targets,old_loss,grads,epsilon):
    new_out = model(data+epsilon*grads)
    new_loss = F.cross_entropy(new_out, targets)
    diff_loss = new_loss - old_loss 
    diff_loss_clampled = torch.clamp(diff_loss,min=0)
    return diff_loss_clampled
    
def pairwise_gradient_transfer(data, targets, data_grads_list, old_losses, epsilon=1):
    loss_avg = 0
    new_losses = []
    count = 0
    loss_numpy = np.zeros((perm(args.models_num,2,exact=True),)) 
    for i in range(0,args.models_num):
        for j in range(0,args.models_num):
            if i==j:
                continue
            diff_loss_clampled = compute_loss_increase(models_list[i],data,targets,old_losses[i],data_grads_list[j],epsilon)
            loss_avg = diff_loss_clampled + loss_avg             
            new_losses.append(diff_loss_clampled)
            loss_numpy[count] = diff_loss_clampled.detach()  
            count = count + 1
    mean_loss = loss_avg / count            
    return new_losses,mean_loss, loss_numpy
    
def compute_input_grads(losses_list,data_batch,if_train=True): 
    #recieves a list of lossess for each model and the input data
    #computes the gradient of each loss wrt to the input 
    #returns the input gradients 
    data_grads_list = []
    for i in range(0,args.models_num):
        if if_train:
            data_grads = torch.autograd.grad(losses_list[i], data_batch,create_graph=True,retain_graph=True)[0]
        else:
            data_grads = torch.autograd.grad(losses_list[i], data_batch,create_graph=True,retain_graph=False)[0].detach()
        #print(data_grads*args.fgm_epsilon_training)
        if torch.sum(torch.isnan(data_grads)) > 0:
            print('Nan in grad of model')
            data_grads[data_grads != data_grads] = 0 
        data_grads_list.append(data_grads)         
    return data_grads_list

def compute_data_losses(data_batch, target_batch):
    #recieves a list of models and the data and target for the batch 
    #computes the loss of each model
    #returns losses as pytorch tensors, the outputs as pytorch tensors, the losses as numpy array (to be used for accumulating the loss per epoch)
    losses_list = []
    outputs_list = []
    models_losses = np.zeros((args.models_num,))
    for i in range(0,args.models_num):
        output = models_list[i](data_batch)
        loss = F.cross_entropy(output, target_batch)    
        outputs_list.append(output.detach())
        losses_list.append(loss)   
        models_losses[i] = loss.detach()     
    return losses_list, outputs_list, models_losses

def compute_total_loss(losses_list,transfer_losses=None, gradient_loss =None, sgn_transfer_losses=None):
    #recieves a list of losses (pytorch tensors) for each model and the gradient loss and sums up the total loss 
    #returns the total loss, will be used for backprop 
    loss = losses_list[0]
    for i in range(1,len(losses_list)):
        loss = loss + losses_list[i]  
    loss = loss / len(losses_list)
    loss = args.classify_weight * loss   
    if gradient_loss != None:    
        loss = loss + args.gradient_weight*shifted_relu_loss(gradient_loss,-1)
           
    if transfer_losses != None :
        transfer_loss = transfer_losses[0]
        for i in range(1,len(transfer_losses)):
            transfer_loss = transfer_loss + transfer_losses[i] 
        transfer_loss = transfer_loss/len(transfer_losses)
        loss = loss + args.transfer_loss_weight*transfer_loss
        
    if sgn_transfer_losses != None :
        sgn_transfer_loss = sgn_transfer_losses[0]
        for i in range(1,len(sgn_transfer_losses)):
            sgn_transfer_loss = sgn_transfer_loss + sgn_transfer_losses[i] 
        sgn_transfer_loss = sgn_transfer_loss/len(sgn_transfer_losses)
        loss = loss + args.transfer_loss_sgn_weight*sgn_transfer_loss

    return loss
    
def compute_correct_predictions(outputs_list, target_batch):   
    #takes the output and target and returns an array of correct predictions for each model 
    #used to test and to accumulate correct predictions across all batches 
    correct_arr = np.zeros((args.models_num,)) 
    for i in range(0,args.models_num):
        output = F.log_softmax(outputs_list[i], dim=1)
        pred1 = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_arr[i] = pred1.eq(target_batch.view_as(pred1)).sum().item()
    return correct_arr              

def create_models():
    #returns a list of models 
    models_list = []
    for i in range(0,args.models_num):
        #model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
        if args.arch == 'resnet18':
            model = PreActResNet18()
        else:
            model = resnet.__dict__[args.arch]()
        model.cuda()
        models_list.append(model)
    return models_list  

def train(train_loader, optimizer, epoch):
    print('-' * 89)
    for i in range(0,args.models_num):
        models_list[i].train()

    sum_total_loss = 0
    sum_gradient_loss = 0
    sum_transfer_loss = np.zeros((perm(args.models_num,2,exact=True),)) 
    sum_sgn_transfer_loss = np.zeros((perm(args.models_num,2,exact=True),)) 
    sum_model_losses = np.zeros((args.models_num,))   
    
    for batch_idx, (data, target) in enumerate(train_loader):

        #initialize losses to None ##
        loss_gradient = None 
        transfer_loss = None   
        sgn_transfer_loss = None        
        
        data, target = data.to(device), target.to(device)

        data.requires_grad = True     
        optimizer.zero_grad()
        if data.grad:        
            data.grad.data.zero_()
        losses_list, outputs_list, losses_np = compute_data_losses(data, target)
        data_grads_list = compute_input_grads(losses_list,data)  
        if args.include_transfer_sgn_loss==1 and epoch > args.start_transfer_sgn_loss:
            data_sign_grads_list = compute_input_sgn_grads(data_grads_list)       
                 
        if args.include_gradient_angle and epoch > args.start_gradient_loss and epoch < args.end_gradient_loss:
            loss_gradient_list,loss_gradient = cosine_loss(data_grads_list, data.size(0))
            sum_gradient_loss += loss_gradient.detach()  
                
        if args.include_transfer_loss == 1 and epoch > args.start_transfer_loss:   
            transfer_loss, transfer_losses_avg, transfer_loss_np = pairwise_gradient_transfer(data,target,data_grads_list,losses_list,args.fgm_epsilon_training)   
            sum_transfer_loss += transfer_loss_np 
            
        if args.include_transfer_sgn_loss == 1 and epoch > args.start_transfer_sgn_loss:   
            sgn_transfer_loss, sgn_transfer_losses_avg, sgn_transfer_loss_np = pairwise_gradient_transfer(data,target,data_sign_grads_list,losses_list,args.fgsm_epsilon_training)   
            sum_sgn_transfer_loss += sgn_transfer_loss_np          
            
        total_loss =  compute_total_loss(losses_list,transfer_loss, loss_gradient,sgn_transfer_loss)
        total_loss.backward()

        sum_total_loss += total_loss.detach()
        sum_model_losses += losses_np
    
        optimizer.step()
        if args.lr_sched == 'cycle':        
            scheduler.step()
        if (batch_idx != 0) and (batch_idx % args.log_interval == 0):
            print('current lr: '+str(scheduler.get_last_lr()))
            cur_loss = sum_total_loss.item() / args.log_interval
            cur_loss_models = sum_model_losses / args.log_interval
            cur_transfer_loss = sum_transfer_loss / args.log_interval 
            cur_sgn_transfer_loss = sum_sgn_transfer_loss / args.log_interval  
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)] | total Loss: {:.6f}\n'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cur_loss))
            if args.include_gradient_angle and epoch > args.start_gradient_loss and epoch < args.end_gradient_loss:
                cur_loss_gradients = sum_gradient_loss.item() / args.log_interval
                print('gradients loss: {:.6f}\n'.format(cur_loss_gradients))
            for i in range(0,args.models_num):                
                print('model{} loss: {:.6f}\n'.format(i,cur_loss_models[i]))
            if args.include_transfer_loss and epoch > args.start_transfer_loss:
                print('Transfer loss: {:.8f}\n'.format(np.mean(cur_transfer_loss)))
            if args.include_transfer_sgn_loss and epoch > args.start_transfer_sgn_loss:
                print('Transfer sign loss: {:.8f}\n'.format(np.mean(cur_sgn_transfer_loss)))
            
            sum_total_loss = 0 
            sum_model_losses = np.zeros((args.models_num,)) 
            sum_gradient_loss = 0
            sum_transfer_loss = np.zeros((perm(args.models_num,2,exact=True),)) 
            sum_sgn_transfer_loss = np.zeros((perm(args.models_num,2,exact=True),))           
            print('-' * 50)
            
def test(test_loader, split_type):
    for i in range(0,args.models_num):
        models_list[i].eval()
    test_loss_tot = 0
    test_loss_models = np.zeros((args.models_num,))
    test_loss_gradients = 0
    test_loss_transfer = 0
    test_loss_sgn_transfer = 0 
    
    correct = np.zeros((args.models_num,))
    batches = 0 
    for data, target in test_loader:
        batches = batches + 1 
        data, target = data.to(device), target.to(device)
        data.requires_grad = True    
        for i in range(0,args.models_num):        
            models_list[i].zero_grad()
        
        losses_list, outputs_list, losses_np = compute_data_losses(data, target)
        data_grads_list = compute_input_grads(losses_list,data,if_train=False)
          
        loss_gradient = None
        transfer_loss = None
        sgn_transfer_loss = None

        with torch.no_grad():
            if args.include_transfer_sgn_loss:
                data_sign_grads_list = compute_input_sgn_grads(data_grads_list) 
            if args.include_gradient_angle == 1 and epoch > args.start_gradient_loss:
                loss_gradient_list,loss_gradient = cosine_loss(data_grads_list, data.size(0))
                test_loss_gradients += loss_gradient.data
            if args.include_transfer_loss == 1 and epoch > args.start_transfer_loss:   
                transfer_loss,transfer_loss_avg,_ = pairwise_gradient_transfer(data,target,data_grads_list,losses_list,args.fgm_epsilon_training)
                test_loss_transfer += transfer_loss_avg.data
            if args.include_transfer_sgn_loss == 1 and epoch > args.start_transfer_sgn_loss:   
                sgn_transfer_loss,sgn_transfer_loss_avg,_ = pairwise_gradient_transfer(data,target,data_sign_grads_list,losses_list,args.fgsm_epsilon_training)
                test_loss_sgn_transfer += sgn_transfer_loss_avg.data      
            
        total_loss =  compute_total_loss(losses_list,transfer_loss, loss_gradient, sgn_transfer_loss)
        test_loss_tot += total_loss.detach()          
        correct += compute_correct_predictions(outputs_list, target)
        test_loss_models += losses_np

    test_loss_models /= batches
    test_loss_models_mean = np.mean(test_loss_models)    
    mean_correct = np.mean(correct)
    test_loss_gradients /= batches
    test_loss_transfer /= batches
    test_loss_sgn_transfer /= batches
    
    test_loss_tot /= batches

    print(split_type+' set: tot loss: {:.4f}, Average model loss: {:.4f}, Average Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss_tot, test_loss_models_mean, mean_correct, len(test_loader.dataset),
        100. * mean_correct / len(test_loader.dataset)))
    if args.include_gradient_angle == 1 and epoch < args.end_gradient_loss:
        print(split_type+' set: gradient angle loss: {:.4f}'.format(test_loss_gradients))
    if args.include_transfer_loss == 1:
        print(split_type+' set: gradient transfer avg loss: {:.8f}'.format(test_loss_transfer))
    if args.include_transfer_sgn_loss == 1:
        print(split_type+' set: gradient sign transfer avg loss: {:.8f}'.format(test_loss_sgn_transfer))

    return test_loss_tot,mean_correct  

def save_models(name):
    for i in range(0,args.models_num):
        torch.save(models_list[i].state_dict(),args.save_dir+'/model'+str(i)+args.save_name+'_'+name+'.pt')

   
transform=transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor()
    ]), download=True)

print(len(dataset1))
main_train_set, val_set = torch.utils.data.random_split(dataset1, [len(dataset1)-5000, 5000],torch.Generator().manual_seed(args.val_seed))
train_loader = torch.utils.data.DataLoader(main_train_set,**kwargs)
val_loader = torch.utils.data.DataLoader(val_set,**kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=args.batch_size, shuffle=False, pin_memory=True)

models_list = create_models()

params = []
for i in range(0,args.models_num):
    params = params + list(models_list[i].parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Model total parameters:', total_params)

lr_steps = args.epochs*len(train_loader)
print(lr_steps)
if args.optimizer_type == 'sgd':
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lr_sched == 'cycle':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.lr_max,step_size_up=lr_steps/2, step_size_down=lr_steps/2)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
elif args.optimizer_type == 'adam':
    optimizer = optim.Adam(params, lr=args.adam_lr, weight_decay=args.weight_decay)  
    if args.lr_sched == 'cycle':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.adam_lr, args.adam_max_lr,cycle_momentum=False,step_size_up=lr_steps/2, step_size_down=lr_steps/2)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
#scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

best_loss_val = 1000000
best_blackboc_acc_subset = 0 


log_file_val = open(args.save_dir+'/log_file_subset_acc.txt','w') 
for epoch in range(1, args.epochs + 1):
    train(train_loader, optimizer, epoch)
    loss_test,acc_test = test(test_loader, 'test')
    loss_val,acc_val = test(val_loader, 'val')
    
    if epoch % args.tranfser_check_interval == 0:
        nonadv_acc_subset = test_acc_allmodels(models_list, device, val_loader)
        if args.attack_type_check == 'pgd':
            transferable_adv_acc_subset = blackbox_allmodels(models_list, device, val_loader, 0.031, 'pgd',0.0078,7)
            print('Val subset: average blackbox PGD: {:.4f}\n'.format(transferable_adv_acc_subset))
        else:
            transferable_adv_acc_subset1 = blackbox_allmodels(models_list, device, val_loader, args.fgm_epsilon, 'fgm')
            transferable_adv_acc_subset2 = blackbox_allmodels(models_list, device, val_loader, args.fgsm_epsilon, 'fgsm')
            transferable_adv_acc_subset = (transferable_adv_acc_subset1+transferable_adv_acc_subset2)/2
            print('Val subset: average blackbox FGM: {:.4f}\n'.format(transferable_adv_acc_subset1))
            print('Val subset: average blackbox FGSM: {:.4f}\n'.format(transferable_adv_acc_subset2))
            print('Val subset: average non adv accuracy: {:.4f}\n'.format(nonadv_acc_subset))
            
        log_file_val.write(str(nonadv_acc_subset) + ', '+ str(transferable_adv_acc_subset) + '\n') 
        log_file_val.flush()
        if transferable_adv_acc_subset > best_blackboc_acc_subset:  
            if args.save_model:
                save_models('best_bb')
            print('Saving model (new best blackbox acc)')   
            best_blackboc_acc_subset = transferable_adv_acc_subset
    if loss_val < best_loss_val:
        if args.save_model:
            save_models('best')
            print('Saving model (new best validation)')
        best_loss_val = loss_val 
    if epoch%2 == 0:
        save_models('interval') 
    #scheduler.step()
    print('-' * 50) 

if args.save_model:
    save_models('end')
    print('Saving model (end)')


