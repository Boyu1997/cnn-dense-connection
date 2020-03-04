# CNN Dense Connection
This project is an inquiry into DenseNet and CondenseNet, specifically, the effect of having dense connections. The project develops a network class with a configurable rate of dense connection, as well as an empirical analysis on the effect of dense connection on network performance and training time.

## Directory Tree
```bash
├── evaluation
│   ├── model_eval.py
│   └── model_select.py # plot performance figure
├── model
│   ├── architectures.py
│   ├── densenet.py
│   ├── helpers.py
│   └── layers.py
├── data.py
├── experiment.sh
└── main.py # entry point from network training
```
