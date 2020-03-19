import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse


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


def plot_time(plot_title, directory, filename_list, filename_suffix):

    training_time_list = []

    # get data from file
    for filename in filename_list:
        f = open('{:s}/{:s}{:s}'.format(directory, filename, filename_suffix),'r')
        data = json.load(f)
        f.close()
        training_time_list.append(data['training_time'])

    # data processing
    training_time_list = np.array(training_time_list)
    X = range(len(filename_list))
    y_mean = np.mean(training_time_list, axis=1)
    y_std = np.std(training_time_list, axis=1)

    # plot figure
    plt.plot(X, y_mean, label='mean',
             marker="x", markeredgecolor='k', markeredgewidth=3, markersize=10)
    plt.fill_between(X, y_mean-1.96*y_std, y_mean+1.96*y_std, label='95% CI',
                     alpha=0.3, color='r', zorder=1)
    plt.title("{:s} Training Time".format(plot_title))
    plt.xticks(X, filename_list, rotation=10)
    plt.ylabel('training time per epoch (s)')
    plt.ylim(30, 40)
    plt.legend(loc=4, prop={'size': 10})
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

    plot_time(plot_title=args.plot_title,
              directory=args.directory,
              filename_list=filename_list,
              filename_suffix=args.filename_suffix)
