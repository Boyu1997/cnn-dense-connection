
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def id_to_num_of_connections(id):
    binary = '{0:010b}'.format(id)
    sum = 0
    for char in binary:
        sum += int(char)
    return sum


results = [[] for _ in range(11)]

for id in range(1024):
    f = open('./save/result/{:s}.json'.format(str(id)),'r')
    data = json.load(f)
    f.close()

    count = id_to_num_of_connections(id)
    results[count].append(data['validation_accuracy'][-1])

X_mean = range(len(results))
y_mean = [sum(result)/len(result) for result in results]

X_std = range(1, len(results)-1)
y_95_up = []
y_95_low = []
for i in range(1, len(results)-1):
    sum = 0
    for num in results[i]:
        sum += (num - y_mean[i])**2
    std = (sum/len(results[i]))**0.5
    y_95_up.append(y_mean[i] + 1.96*std)
    y_95_low.append(y_mean[i] - 1.96*std)

plt.plot(X_mean, y_mean, label='mean',
         marker="x", markeredgecolor='k', markeredgewidth=3, markersize=10)
plt.fill_between(X_std, y_95_up, y_95_low, label='95% CI', alpha=0.3, color='r', linewidth=0, zorder=1)
plt.plot(X_std, y_95_up, 'r')
plt.plot(X_std, y_95_low, 'r')
plt.title('Model Performance')
plt.xlabel('number of dense connections')
plt.xticks(X_mean, range(11))
plt.ylabel('accuracy')
plt.ylim(0.2, 0.8)
plt.legend(loc=4, prop={'size': 10})
plt.show()
