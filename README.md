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
