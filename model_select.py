import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def moving_average(a, n=10):
    df = pd.DataFrame({'a': a})
    df['m'] = df['a'].rolling(window=n).mean()
    return df['m']


f = open('./save/densenet/train.json','r')
densenet_data = json.load(f)
f.close()

f = open('./save/densenet2/train.json','r')
densenet2_data = json.load(f)
f.close()

# plot loss plot
x = np.arange(240)
acc_x = np.arange(20, 240)
fig = plt.figure()

ax = plt.subplot(221)
ax.plot(x, densenet_data['training_loss'][:240], label='training loss')
ax.plot(x, densenet_data['validation_loss'][:240], label='validation loss')
ax.plot(x, moving_average(densenet_data['validation_loss'][:240], 20), label='validation moving average')
plt.title('DenseNet Loss Over Epoch')
plt.ylim(0, 0.8)
ax.legend(loc=1, prop={'size': 9})

ax = plt.subplot(222)
ax.plot(acc_x, [0.875 for _ in range(220)], 'r--', label='87.5% accuracy')
ax.plot(acc_x, densenet_data['training_accuracy'][20:240], label='training accuracy')
ax.plot(acc_x, densenet_data['validation_accuracy'][20:240], label='validation accuracy')
plt.title('DenseNet Loss Over Epoch')
plt.ylim(0.8, 1)
ax.legend(loc=4, prop={'size': 9})

ax = plt.subplot(223)
ax.plot(x, densenet2_data['training_loss'][:240], label='training loss')
ax.plot(x, densenet2_data['validation_loss'][:240], label='validation loss')
ax.plot(x, moving_average(densenet2_data['validation_loss'][:240], 20), label='validation moving average')
plt.title('DenseNet2 Loss Over Epoch')
plt.ylim(0, 0.8)
ax.legend(loc=1, prop={'size': 9})

ax = plt.subplot(224)
ax.plot(acc_x, [0.875 for _ in range(220)], 'r--', label='87.5% accuracy')
ax.plot(acc_x, densenet2_data['training_accuracy'][20:240], label='training accuracy')
ax.plot(acc_x, densenet2_data['validation_accuracy'][20:240], label='validation accuracy')
plt.title('DenseNet2 Loss Over Epoch')
plt.ylim(0.8, 1)
ax.legend(loc=4, prop={'size': 9})

plt.show()
