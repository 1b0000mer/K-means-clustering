import matplotlib.pyplot as plt
from itertools import cycle
import function as f
from sklearn.datasets import make_blobs
from itertools import cycle
import numpy as np
import pandas as pd
import os

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

#samples data
#Generate sample data
n = 100 # number elements for each cluster
dim = 2 # 2-D data point aka features
k = 3
spread = .5 #0-1

#size = (k,dim)
#min_value = 2
#max_value = 10
#centers = np.random.randint(min_value, max_value, size=size)

centers = [[1,3], [1,4], [1,2]] # 3 clusters overlapped
#centers = [[1,1], [4,4], [7,7]] # 3 clusters separated

data, labels_true = make_blobs(n_samples=n*3, centers=centers, cluster_std=spread, random_state=0)

fig1 = plt.figure(1)
plt.scatter(data[:,0], data[:,1])

fig2 = plt.figure(2, figsize=(9,7))
plt.title('Data with true labels')
for i, color in zip(range(k), colors):
    k_members = labels_true == i
    cluster_center = centers[i]
    plt.plot(data[k_members, 0], data[k_members, 1], color + '.') #plot members
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

plt.show()

result, new_centers, labels_learned, figures = f.new_kmeans(data, dim, k, rand_centers=False, gif=False)
f1 = f.F1_score(labels_true, labels_learned)


fig3 = plt.figure(3)
for i, color in zip(range(k), colors):
    temp = np.array(result[i])
    plt.scatter(temp[:,0], temp[:,1])

fig4 = plt.figure(4, figsize=(9,7))
plt.title('Data with learned labels. F1 score: %.4f' % f1)
for i, color in zip(range(k), colors):
    k_members = labels_learned == i
    cluster_center = new_centers[i]
    plt.plot(data[k_members, 0], data[k_members, 1], color + '.') #plot members
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

plt.show()

#figures.insert(0, fig2)
#figures.append(fig4)
#f.make_gif(figures, 'test') #all figures must be same size!