# Low-Rank Compression of Deep Neural Networks using SVD

This repository contains the implementation of **Project 6** for the course **Computational Data Mining**.

## Project Overview
The goal of this project is to study **low-rank compression of deep neural networks** using **Singular Value Decomposition (SVD)**.  
A pretrained ResNet20 model on the CIFAR-10 dataset is analyzed and partially compressed without changing the overall architecture.

The project includes:
- Spectral analysis of network weights
- Low-rank approximation of a linear layer using SVD
- Evaluation of accuracy before and after compression
- Fine-tuning to recover lost accuracy
- CPU inference latency analysis
- Accuracy vs compression rate visualization

## Files
- `NoteBook.ipynb`: Main notebook containing all experiments and analysis

## Dataset
The CIFAR-10 dataset is automatically downloaded using PyTorch and is **not included** in this repository.

## Course Information
- Course: Computational Data Mining
- Project: Project 6
