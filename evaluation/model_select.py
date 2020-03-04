import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def moving_average(a, n=10):
    df = pd.DataFrame({'a': a})
    df['m'] = df['a'].rolling(window=n).mean()
    return df['m']

def plot_loss(ax, data, skip_first=20, title=None):
    l = len(data['training_loss'])
    x = np.arange(l)
    ax.plot(x, [0.5 for _ in range(l)], 'r--', label='loss=0.5')
    ax.plot(x, data['training_loss'], label='training loss')
    ax.plot(x, data['validation_loss'], label='validation loss')
    ax.plot(x, moving_average(data['validation_loss'], 20), label='validation moving average')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(skip_first, l)
    plt.ylim(0, 0.8)
    ax.legend(loc=4, prop={'size': 8})


def plot_accuracy(ax, data, skip_first=20, title=None):
    l = len(data['training_accuracy'])
    x = np.arange(l)
    ax.plot(x, [87.5 for _ in range(l)], 'r--', label='87.5% accuracy')
    ax.plot(x, [i*100 for i in data['training_accuracy']], label='training accuracy')
    ax.plot(x, [i*100 for i in data['validation_accuracy']], label='validation accuracy')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.xlim(skip_first, l)
    plt.ylim(80, 100)
    ax.legend(loc=4, prop={'size': 8})


# data_path_list = [
#     'densenet-original',
#     'cbr=0.2-reduction=0.5',
#     'cbr=0.4-reduction=0.5',
#     'cbr=0.6-reduction=0.5'
# ]

data_path_list = [
    'condensenet-original',
    'cbr=0.8-reduction=0.1',
    'cbr=0.6-reduction=0.1',
    'cbr=0.4-reduction=0.1'
]


fig = plt.figure()
fig.suptitle('Traing Result for CondenseNet and CondenseNet Variations Under Cosine Decay Learning Rate')
subplot_idx = 1

for data_path in data_path_list:
    f = open('../save/save-cosine-lr/{:s}/train.json'.format(data_path),'r')
    data = json.load(f)
    f.close()

    ax = plt.subplot(2, 4, subplot_idx)
    plot_loss(ax, data, title=data_path)

    ax = plt.subplot(2, 4, subplot_idx+4)
    plot_accuracy(ax, data)
    subplot_idx += 1

plt.show()
