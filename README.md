# CNN Dense Connection
This project is an inquiry into DenseNet and CondenseNet, specifically, the effect of having dense connections. The project develops a network class with a configurable rate of dense connection, as well as an empirical analysis on the effect of dense connection on network performance and training time.


## Contents

1. [Preliminary](#preliminary)
2. [Usage](#usage)
3. [Results](#results)
4. [Acknowledgement](#acknowledgement)


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
│   ├── model_eval.py
│   └── model_select.py # plot performance figure
├── model
│   ├── architectures.py
│   ├── densenet.py
│   └── helpers.py
├── data.py
├── experiment.sh
└── main.py # entry point from network training
```


## Results
TBD


## Acknowledgement

- [Python3](https://www.python.org)
- [PyTorch](http://pytorch.org)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [CondenseNet](https://arxiv.org/abs/1711.09224)
