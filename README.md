# Deep One-Class Classification  
**Final Project – Unsupervised Machine Learning (Spring 2025)**  
By Vedanshi Shah, Abbie Murphy and Jimmy Kong

---

## 🔍 Project Overview

This project explores the **Deep One-Class Classification** method for anomaly detection in high-dimensional data. We analyze baseline approaches and improve performance by experiment with various preproceesing, during processing, and post-processing adjustments. 

The goal is to explore the challenges in unsupervised anomaly detection and improve performance by leveraging different unsupervised algorithms. The project includes visualizations of results and simulated anomaly detection predictions.

---

## 🌐 Live Demo

🖥️ [View the interactive project website](https://vedanshishah7.github.io/unsupervised_ml_final_project/)

Key features include:
- AUC comparison charts for three datasets: MNIST, CIFAR-10, and Fashion-MNIST.
- Simulated anomaly detection predictions.
- Interactive visualization using Chart.js.
- AUC based visualizations.

---
## 🔬 Experiments on Original Code
`Deep_SVDD-PyTorch-FPSVDD` includes a copy of the authors original [Deep SVDD Github repository implemented in PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch) with two optimizations that we added for experimentation. To run both of these variations, the user must navigate to the aforementioned directory, then navigate to 'src' before running the appropriate experiments.

### Post-Processing
For post processing we implemented z-score normalization on the resulting anomaly scores. This will run each time that experiments are run on a class. 

An example of how to run on the MNIST dataset:
```
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3;
```

### Feature Patching
Feature patching is implemented specifically for the Cifar10 dataset. This variation can be run by setting the `--use_fpsvdd` flag.

```
python main.py cifar10 cifar10_LeNet ../log/cifar10_fpsvdd/cifar10_fp2 ../data --objective one-class --lr 0.0001 --device "cuda" --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 2 --use_fpsvdd;
```
---
## 📂 Document Links
- [Final Report](https://docs.google.com/document/d/1dwaurbHzPVQcYql9VhDIC10omdElVdYjvM4fZ-zNziI/edit?usp=drive_link)
- [Presentation](https://docs.google.com/presentation/d/17Nf1NvzlN5sNUOu8Ogicwa4o2qFJa9s1CTXdhLQ9P84/edit?usp=drive_link)

