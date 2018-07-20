# coding=UTF-8
import os
import os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from PIL import Image
from load.load_mnist import load
from sklearn.datasets import load_iris

file_path = os.path.dirname(os.path.abspath(__file__))
file_name = file_path + "/data_iris/score_iris.csv"
gmm_file_name = file_path + "/data_iris/gmm_score_iris.csv"

iris_k_means = np.array(pd.read_csv(file_name, usecols=[2]))
iris_gmm = np.array(pd.read_csv(gmm_file_name, usecols=[2]))

# df = pd.DataFrame(np.c_[iris_k_means, iris_gmm]).rename(columns={0: 'K_means', 1: 'GMM'})
d1 = list(iris_k_means.reshape([100]))
print(d1)
d2 = list(iris_gmm.reshape(100))
data = (d1, d2)
print(data)

fig = plt.figure()
ax = fig.add_subplot(111)


# データをセット
bp = ax.boxplot(data)

# 横軸のラベルの設定
ax.set_xticklabels(['K Means', 'GMM'])

# グリッド線を表示
plt.grid()

# 横軸のラベルを設定
plt.xlabel('')

# 縦軸のラベルを設定
plt.ylabel('entropy')

# タイトルを設定
plt.title('box plot')

# 縦軸の範囲を設定
plt.ylim([0, 1])

# 箱ひげ図の表示
plt.show()
"""
fig, axes = plt.subplots(1, 1)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for(ax, (key, group)) in zip(axes, df.groupby(pd.Grouper(level=0, freq='M'))):
    ax = group.plot.box(ax=ax)
    ax.set_ylabel('entropy')
    ax.set_title(key, fontsize=8)

fig.show()
"""
