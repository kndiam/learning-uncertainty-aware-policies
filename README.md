# adaptive-coverage-policies

This repository contains the code for reproducing the experiments and figures presented in the paper [Adaptive Coverage Policies in Conformal Prediction](https://arxiv.org/abs/2510.04318).

## Organization

The repository contains two main folders, each corresponding to one of the experiments presented in the paper: one for classification and one for regression. Each folder is self-contained.

## Instructions

### Classification

1. (*Optional*) Run the `model_train.ipynb` notebook to re-train the model f. This will download and store the CIFAR-10 dataset in the `data/` subfolder.

2. Execute the `train-coverage-policy.ipynb` notebook to reproduce the experiments corresponding to Figures 1, 2, and 3. The notebook will load the CIFAR-10 dataset from the `data/` subfolder (downloading it if necessary).

3. Execute the `selection-lambda.ipynb` notebook to reproduce the experiment corresponding to Figure 4.
  
4. The resulting plots are stored in the `plots/` subfolder.

### Regression

1. Execute the `regression.ipynb` notebook to reproduce all regression experiments from the paper.

2. The resulting plots are stored in the `plots/` subfolder.