# "What’s in the box?!": Deflecting Adversarial Attacks by Randomly Deploying Adversarially-Disjoint Models #
- Code for the paper: ["What's in the box?!": Deflecting Adversarial Attacks by Randomly Deploying Adversarially-Disjoint Models](https://arxiv.org/pdf/2102.05104.pdf) 
- Authors: [Sahar Abdelnabi](https://scholar.google.de/citations?user=QEiYbDYAAAAJ&hl=en), [Mario Fritz](https://cispa.saarland/group/fritz/)

## Abstract ## 
Machine learning models are now widely deployed in real-worldapplications. However, the existence of adversarial examples hasbeen long considered a real threat to such models. While numerousdefenses aiming to improve the robustness have been proposed,many have been shown ineffective. As these vulnerabilities arestill nowhere near being eliminated, we propose an alternativedeployment-based defense paradigm that goes beyond the tradi-tional white-box and black-box threat models. Instead of trainingand deploying a single partially-robust model, one could train aset of same-functionality, yet,adversarially-disjointmodels withminimal in-between attack transferability. These models could thenberandomly and individuallydeployed, such that accessing one ofthem minimally affects the others. Our experiments on CIFAR-10and a wide range of attacks show that we achieve a significantlylower attack transferability across our disjoint models compared toa baseline of ensemble diversity. In addition, compared to an adver-sarially trained set, we achieve a higher average robust accuracywhilemaintainingthe accuracy of clean examples.

<p align="center">
<img src="https://github.com/S-Abdelnabi/AdversariallyDisjoint/blob/main/teaser.PNG" width="600">
</p>

- - -

## Enviroment ##
- Main requirements:
	- Python 3.8.5
	- PyTorch 1.6.0
- To set it up: 
```javascript
conda env create --name AdvDisjoint --file=environment.yml
```
- - -
## Pre-trained models ##

You can find pre-trained models for the different sets (ranging from 3 to 8 models) [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/LSfHBZfxozQAWAm).

- - -

## Training ##
Training sets of adversarially-disjoint models, dataset: CIFAR-10, architecture:  PreAct ResNet18

### Set of 3 models ###
```javascript
python main.py --optimizer_type sgd --lr 0. --lr_max 0.2 --lr_sched cycle --epochs 76 --models_num 3 --seed 1002 --tranfser_check_interval 4 --attack_type_check pgd --include_gradient_angle 1 --gradient_weight 0.4 --start_gradient_loss 0 --end_gradient_loss 5 --include_transfer_loss 1 --start_transfer_loss 1 --include_transfer_sgn_loss 1 --transfer_loss_weight 0.5 --start_transfer_sgn_loss 2 --transfer_loss_sgn_weight 0.4 --fgm_epsilon_training 6 --arch resnet18 --batch_size 120
```
### Set of 4 models ###
```javascript
python main.py --optimizer_type sgd --lr 0. --lr_max 0.2 --lr_sched cycle --epochs 76 --models_num 4 --seed 1002 --tranfser_check_interval 4 --attack_type_check pgd --include_gradient_angle 1 --gradient_weight 0.4 --start_gradient_loss 0 --end_gradient_loss 5 --include_transfer_loss 1 --start_transfer_loss 1 --include_transfer_sgn_loss 1 --transfer_loss_weight 0.5 --start_transfer_sgn_loss 2 --transfer_loss_sgn_weight 0.4 --fgm_epsilon_training 6 --arch resnet18 --batch_size 106
```
### Set of 5 models ###
```javascript
python main_sample.py --optimizer_type sgd --lr 0. --lr_max 0.2 --lr_sched cycle --epochs 100 --models_num 5 --seed 1002 --tranfser_check_interval 4 --attack_type_check pgd --include_gradient_angle 1 --gradient_weight 0.4 --start_gradient_loss 0 --end_gradient_loss 5 --include_transfer_loss 1 --start_transfer_loss 0 --include_transfer_sgn_loss 1 --transfer_loss_weight 0.8 --start_transfer_sgn_loss 0 --transfer_loss_sgn_weight 0.8 --fgm_epsilon_training 6 --arch resnet18 --batch_size 106
```
### Set of 6 models ###
```javascript
python main__sample.py --optimizer_type sgd --lr 0. --lr_max 0.2 --lr_sched cycle --epochs 120 --models_num 6 --seed 1002 --tranfser_check_interval 4 --attack_type_check pgd --include_gradient_angle 1 --gradient_weight 0.4 --start_gradient_loss 0 --end_gradient_loss 5 --include_transfer_loss 1 --start_transfer_loss 0 --include_transfer_sgn_loss 1 --transfer_loss_weight 0.8 --start_transfer_sgn_loss 0 --transfer_loss_sgn_weight 0.8 --fgm_epsilon_training 6 --arch resnet18 --batch_size 106 
```
### Set of 7 models ###
```javascript
python main_sample.py --optimizer_type sgd --lr 0. --lr_max 0.1 --lr_sched cycle --epochs 120 --up_epochs 30 --down_epochs 90 --models_num 7 --seed 1002 --tranfser_check_interval 4 --attack_type_check pgd --include_gradient_angle 1 --gradient_weight 0.4 --start_gradient_loss 0 --end_gradient_loss 5 --include_transfer_loss 1 --start_transfer_loss 0 --include_transfer_sgn_loss 1 --transfer_loss_weight 0.8 --start_transfer_sgn_loss 0 --transfer_loss_sgn_weight 0.8 --fgm_epsilon_training 6 --arch resnet18 --batch_size 90
```
### Set of 8 models ###
```javascript
python main_sample.py --optimizer_type sgd --lr 0. --lr_max 0.1 --lr_sched cycle --epochs 120 --up_epochs 30 --down_epochs 90 --models_num 8 --seed 1002 --tranfser_check_interval 4 --attack_type_check pgd --include_gradient_angle 1 --gradient_weight 0.4 --start_gradient_loss 0 --end_gradient_loss 5 --include_transfer_loss 1 --start_transfer_loss 0 --include_transfer_sgn_loss 1 --transfer_loss_weight 0.8 --start_transfer_sgn_loss 0 --transfer_loss_sgn_weight 0.8 --fgm_epsilon_training 6 --arch resnet18 --batch_size 75 
```
- - -
## Evaluation ##

### Clean accuracy ###
```javascript
python evaluate_nonadv_acc.py --models_num <num> --save_dir <checkpt_folder>  --arch resnet18 --batch_size 128 --checkpt _best
```

### Blackbox attacks across models and average of sets ###
#### FGSM ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num> --save_dir <checkpt_folder> --attack_type fgsm --epsilon 0.031 --arch resnet18 --batch_size 128 --checkpt _best
```
#### FGM ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num>  --save_dir <checkpt_folder> --attack_type fgm --epsilon 1 --arch resnet18 --batch_size 128 --checkpt _best
```
#### R-FGSM ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num> --save_dir <checkpt_folder> --attack_type rfgsm --epsilon 0.031 --arch resnet18 --batch_size 128 --checkpt _best
```
#### PGD ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num> --save_dir <checkpt_folder> --attack_type pgd --epsilon 0.031 --alpha 0.0078 --bim_itrs <iters> --arch resnet18 --batch_size 128 --checkpt _best
```
#### MI-FGSM ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num> --save_dir <checkpt_folder> --attack_type mifgsm --epsilon 0.031 --alpha 0.0031 --mi_itrs 20 --arch resnet18 --batch_size 128 --checkpt _best
```
#### C&W ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num> --save_dir <checkpt_folder> --attack_type cw --cw_c 1.0 --arch resnet18 --batch_size 128 --checkpt _best --cw_conf <conf>
```
#### EAD ####
```javascript
python evaluate_blackbox_across_models.py --models_num <num> --save_dir <checkpt_folder> --attack_type ead --arch resnet18 --batch_size 128 --checkpt _best --ead_conf <conf> 
```
### Ensemble Attacks ###
Attacks computed using a subset of models, and evaluated on the rest of the set. Running the following file will output the results in 'ens_out_#num_models.txt'

```javascript
python evaluate_ens_attacks.py --models_num <num> --save_dir <checkpt_folder> --attack_type fgsm --epsilon 0.031 --batch_size 128 --checkpt _best
```

```javascript
python evaluate_ens_attacks.py --models_num <num> --save_dir <checkpt_folder>  --attack_type pgd --epsilon 0.031 --alpha 0.0078 --bim_itrs 20 --arch resnet18 --batch_size 128 --checkpt _best
```


