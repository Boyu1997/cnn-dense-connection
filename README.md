
# CNN Dense Connection
This project is an inquiry into DenseNet and CondenseNet, specifically, the effect of having dense connections. The project develops a network class with a configurable rate of dense connection, as well as an empirical analysis on the effect of dense connection on network performance and training time.


## Contents

1. [Preliminary](#preliminary)
2. [Usage](#usage)
3. [Results](#results)
4. [Playground](#playground)
5. [Acknowledgement](#acknowledgement)


## Preliminary

### DenseNet
DenseNet is a compact network architecture, it introduces dense connections between network layers. Dense connection constructs an architecture with more connectivity compare to a traditional sequential network with similar parameter size.

### CondenseNet
CondenseNet architecture is one step further on DenseNet. In addition to dense connection with in the same block, CondenseNet keeps doing dense connection and re-use feature over the entire network.


## Usage

### Train
Training under default configuration
```
python main.py
```

Options
```BASH
# model architecture setting
--cross_block_rate=0.5   # dense connection rate between dense blocks, use 0 for original DenseNet and 1 for CondenseNet
--end_block_reduction_rate=0.5   # feature channel reduction rate at the end of each dense block

# model configuration
--stages='10-10-10'   # layers for each stage
--growth='8-16-32'   # growth rate for each stage
--group_1x1=4   # number of 1-by-1 group convolution
--group_3x3=4   # number of 3-by-3 group convolution
--bottleneck=4   # bottleneck
--optimizer='sgd'   # optimizer function
--lr=0.1   # learning rate
--scheduler='cos'   # learning rate scheduler
--ep=120   # number of epoch to train
--bsize=512   # training batch size
--one_batch   # train only the first batch of every epoch, for fast local code testing

# parallel gpu setting
--parallel   # use parallel gpu training
--n_gpu=4   # number of gpu used for parallel

# save model
--save_folder='default'   # folder to save model and training details
```

### Evaluation
TBD

### Directory Tree
```bash
├── evaluation
│   ├── model_eval.py
│   └── model_select.py # plot performance figure
├── example
│   ├── result # json files, result for models in this report
│   └── example.ipynb # python notebook, demo training and cresult
├── model
│   ├── architectures.py
│   ├── densenet.py
│   └── helpers.py
├── playground
│   └── README.md # information on interactive playground web app
├── augment.py # data augmentation function
├── data.py # load data to pytorch dataloader
├── experiment.sh # bash script, train and test all models in this report
└── main.py # entry point from network training
```


## Playground

Interactive web application to show how dense connections impact model performance using a 6-layers CondenseNet architecture. Access the application here: [playground.condensenet.boyu.io](http://playground.condensenet.boyu.io/)


## Results

| Method                | Depth | Params | C10 | C10+  |
|-----------------------|-------|--------|-----|-------|
| DenseNet              | 30    | 0.24M  | -   | 15.45 |
| DenseNet (cbr=0.2)    | 30    | 0.26M  | -   | 14.18 |
| DenseNet (cbr=0.4)    | 30    | 0.29M  | -   | 14.80 |
| DenseNet (cbr=0.6)    | 30    | 0.31M  | -   | 15.65 |
| CondenseNet           | 30    | 0.32M  | -   | 13.95 |
| CondenseNet (cbr=0.8) | 30    | 0.29M  | -   | 13.98 |
| CondenseNet (cbr=0.6) | 30    | 0.27M  | -   | 15.06 |
| CondenseNet (cbr=0.4) | 30    | 0.25M  | -   | 14.78 |
| DenseNet              | 78    | 1.09M  | -   | 8.08  |
| DenseNet (cbr=0.2)    | 72    | 1.08M  | -   | 7.99  |
| CondenseNet (cbr=0.8) | 66    | 1.07M  | -   | 7.74  |

\* `C10+` represent cifar-10 dataset with data augmentation.


## Acknowledgement

- [PyTorch](http://pytorch.org)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [CondenseNet](https://arxiv.org/abs/1711.09224)
