# A Black-box Adversarial Attack for Poisoning Clustering
Here we provide the code used to obtain the results proposed in "A Black-box Adversarial Attack for Poisoning Clustering" (Pattern Recognition 2021).

Project structure:

 - `clustorch`, clustering algorithms implementations;
 
 - `comparison/`, comparison between our algorithm and [Chhabra, Roy,and Mohapatra 2019].;
    - `DIGIT`, `MNIST`, `HAND-3-5` and `SEEDS` contain the input and the adversarial dataset 
       used in [Chhabra, Roy,and Mohapatra 2019];
    
 - `experiments/`, source code for reproducing our experimental results (robustness, transferability  and ablation);
    - `classifiers`, pre-trained supervised classifiers used to test the transferability;
    
 - `data/`, dataset used in our tests (FashionMNIST and CIFAR10);
 
 - `src/`, contains the implementation of our threat algorithm, optimization procedure and 
            supervised training code;
    - `classification`, definition and training code of the tested supervised models;
        - `cifar_features`, features extraction for CIFAR10 dataset with ResNet18;
    - `optimizer`, Genetic Algorithm optimizer;
    - `threat/clustering/`, implementation of our threat algorithms (Algorithm 1) for poisoning
     clustering;

## Installation
Import the conda environment:
```bash
$ conda env create -f environment.yml
```
Then, we need to activate the conda env with:
```bash
$ conda activate black-box-poisoning-pr2021
```

## Robustness analysis

We ran the experiments on three real-world datasets: FashionMNIST, CIFAR-10, and 20 Newsgroups. We focused our analysis on both two- and multiple-way clustering problems. For FashionMNIST and 20 Newsgroups, we simulated the former scenario where an attacker wants to perturb samples of one victim cluster $`C_v`$ towards a target cluster $`C_t`$. For CIFAR-10, we allowed the attacker to move samples from multiple-victim clusters towards a target one by simply running our algorithm multiple times with a different victim cluster for each run. 
In the experiments, we chose $`T`$ to contain the $`s|C_v|`$ nearest neighbors belonging to the currently chosen victim cluster, with respect to the centroid of the target cluster. In particular, for FashionMNIST we used 20 different values for s and $`\delta`$, in the intervals [0.01, 0.6] and [0.05, 1] respectively; for CIFAR-10 we used 20 different values for s and $`\delta`$, in the intervals [0.01, 0.6] and [0.01, 1.5] respectively; for 20 Newsgroups we used 15 different values for s and $`\delta`$, in the intervals [0.01, 0.3] and [0.001, 0.3] respectively.

The source code for the experimental setting can be found in folder `black-box-poisoning/experiments/`.

### Run robustness experiments
To replicate the experiments for FashionMNIST, CIFAR10, and 20Newsgroup, you need to run the command:
```bash
$ python experiments/run_fashion.py # FashionMNIST
$ python experiments/run_20news.py # 20newsgroup
$ python experiments/run_cifar.py # cifar10
```
The code we designed is device-agnostic, meaning that the user can set the device over which the computations will be executed. By default we set to `device=cpu`, to run in GPU:
```bash
$ python experiments/run_fashion.py --device='cuda:0'# FashionMNIST
$ python experiments/run_20news.py  --device='cuda:0'# 20newsgroup
$ python experiments/run_cifar.py --device='cuda:0'# cifar10
```

The user may also provide the output directory, where all results will be saved:
```bash
$ python experiments/run_fashion.py --device='cuda:0' --path='./experiments/results/' # FashionMNIST
```
The default output path is set to: `./experiments/results/pr2021/`.

## Transferability to supervised classifiers
We extend the transferability analysis showing that even adversarial samples crafted against clustering algorithms can be successfully transferred to fool supervised models. 
We evaluate the transferability properties of our noise by attacking the K-means++ algorithm on 2000 testing samples taken from labels FashionMNIST (Ankle boot, Dress). 
In particular, we use the crafted adversarial samples to test the robustness of several classification models: a linear and RBF SVM, two random forests with 10 and 100 trees, respectively, and the Carlini \& Wagner (C\&W) deep net. 

To replicate the proposed results, the user should run the command:
```bash
$ python experiments/run_transfer_to_classifiers.py  --device='cuda:0'
```

## Ablation on the clustering similarity measure

We decided to analyze the impact of the clustering similarity function $`\phi`$, with FashionMNIST, and see if there were significant differences among each other. We set $`\phi`$ equals to ARI, AMI, and the negated distance.

In order to replicate the proposed results, the user should run the command:
```bash
$ python experiments/run_ablation.py  --device='cuda:0' --phi='AMI' # AMI curve
$ python experiments/run_ablation.py  --device='cuda:0' --phi='AMI' # ARI curve
$ python experiments/run_ablation.py  --device='cuda:0' --phi='frobenius' # Frobenius curve
```

## Comparison with state-of-the-art
To the best of our knowledge, the only work dealing with adversarial clustering in a black-box way is [Chhabra, Roy, and Mohapatra 2019]. 

Although our algorithm achieves its best performance by moving more samples at once, we could match, or even exceed, the number of spill-over samples achieved in [Chhabra, Roy, and Mohapatra 2019]. Moreover, the results show also that we were able to craft adversarial noise masks $`\vec{\epsilon}`$, which were significantly less detectable in terms of $`\ell_0`$, $`\ell_\infty`$ norms.

To run the code used for comparing the two works, the user should run:

```bash
$ python comparison/comparisonDIGIT.py  --dataset='89' # DIGIT 89
$ python comparison/comparisonDIGIT.py  --dataset='14' # DIGIT 14
$ python comparison/comparisonMNIST14.py # MNIST 14
$ python comparison/comparisonMNIST32.py # MNIST 32
$ python comparison/comparisonSeeds.py # Seeds
$ python comparison/comparisonHand.py # Hands
```


## Cifar10 features extractiong
For clustering CIFAR10 samples, we used a ResNet18 for features extraction, and then clustering algorithms were performed on the resulting embedding space. In `src/classification/cifar_features`, we provide the code for training the net and extract features from CIFAR10.

Train Resnet18:
```bash
$ python src/classification/cifar_features/pytorchcifar/main.py
```

Extract features for CIFAR10:
```bash
$ python src/classification/cifar_features/extract_features.py
```

Build data for robustness analysis:
```bash
$ python src/classification/cifar_features/cifar10.py
```
