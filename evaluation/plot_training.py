import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse

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


densenet_list = [
    'densenet-original',
    'cbr=0.2-reduction=0.5',
    'cbr=0.4-reduction=0.5',
    'cbr=0.6-reduction=0.5'
]

condensenet_list = [
    'condensenet-original',
    'cbr=0.8-reduction=0.1',
    'cbr=0.6-reduction=0.1',
    'cbr=0.4-reduction=0.1'
]

custom_list = [
    'con-cbr=0.2-reduction=0.5',
    'exp-cbr=0.2-reduction=0.5'
]

def plot_training(plot_title, directory, filename_list, filename_suffix):
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle('Traing Result for {:s}'.format(plot_title))
    subplot_idx = 1
    data_count = len(filename_list)

    for filename in filename_list:
        f = open('{:s}/{:s}{:s}'.format(directory, filename, filename_suffix),'r')
        data = json.load(f)
        f.close()

        ax = plt.subplot(2, data_count, subplot_idx)
        plot_loss(ax, data, title=filename)

        ax = plt.subplot(2, data_count, subplot_idx+data_count)
        plot_accuracy(ax, data)
        subplot_idx += 1

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='../save',
        help='path to the save directory'
    )
    parser.add_argument('--filename',
        type=str,
        default='densenet',
        help='serie of filename the ploting data is located'
    )
    parser.add_argument('--filename_suffix',
        type=str,
        default='/train.json',
        help='suffix after filenames'
    )
    parser.add_argument('--plot_title',
        type=str,
        default='DenseNet and DenseNet Variations',
        help='plot title'
    )
    args, _ = parser.parse_known_args()

    if args.filename == 'densenet':
        filename_list= densenet_list
    elif args.filename == 'condensenet':
        filename_list = condensenet_list
    else:
        filename_list = custom_list

    plot_training(plot_title=args.plot_title,
                  directory=args.directory,
                  filename_list=filename_list,
                  filename_suffix=args.filename_suffix)
