import matplotlib.pyplot as plt
from itertools import cycle
import function as f
from sklearn.datasets import make_blobs
from itertools import cycle
import numpy as np
import pandas as pd
import os

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
file_path = os.getcwd() + '\data\customers\Mall_Customers_edited.csv'
features = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

data = f.read_file(file_path, features, 'CustomerID')
features.remove('CustomerID') #index_col does not count
data2, features = f.remove_features(data, features, ['Gender'])
x = np.array(data2.values, dtype=float)
n_features = len(features)
k = 3
#centers=np.array([
#    [19, 13, 3], #young/low_income/low_spendscore
#    [30, 12, 2], 
#    [60, 11, 1],
#    [19, 67, 50], 
#    [30, 66, 49], #mid/mid_income/mid_spendscore
#    [60, 65, 48],
#    [19, 80, 75],
#    [30, 79, 74],
#    [60, 78, 73]]) #eld/hig_income/high_spendscore
#k = len(centers)

#centers=np.array([
#    [19, 65, 70], #young/high_income/high_spendscore
#    [30, 65, 69], #mid/high_income/high_spendscore
#    [60, 65, 68]]) #eld/high_income/high_spendscore
#k = len(centers)

#plot data
fig1 = plt.figure(1)
fig1 = plt.axes(projection='3d')
plt.title('Original')
fig1.set_xlabel('Age')
fig1.set_ylabel('Annual Income (k$)')
fig1.set_zlabel('Spending Score (1-100)')
fig1.scatter3D(x[:,features.index('Age')], x[:,features.index('Annual Income (k$)')], x[:,features.index('Spending Score (1-100)')])

##kmeans from sklearn
from sklearn.cluster import KMeans
clustering = KMeans(k, random_state=None).fit(x)
sk_labels = clustering.labels_
sk_centers = clustering.cluster_centers_



while True:
    result, learned_centers, learned_labels, figures = f.new_kmeans(x, n_features, k, rand_centers=False, gif=False)
    f1 = f.F1_score(sk_labels, learned_labels)
    if (0.8 <= f1 <= 1):
        break
    print ('\nre run, F1=%.4f' %f1, end='\n')

print('F1 score=%.4f' %f1)

#figure 2
a=0
b=1
fig2 = plt.figure(figsize=(9,7))
plt.subplot(2,1,1)
plt.xlabel(features[a])
plt.ylabel(features[b])
plt.title('Relations between '+features[a]+' vs '+features[b])
for i, color in zip(range(k), colors):
    k_members = learned_labels == i
    cluster_center = learned_centers[i]
    plt.plot(x[k_members, a], x[k_members, b], color + '.') #plot members
    plt.plot(cluster_center[a], cluster_center[b], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k
plt.subplot(2,1,2)
plt.xlabel(features[a])
plt.ylabel(features[b])
plt.title('Down is from sklearn. F1=%.4f' %f1, loc='right')
for i, color in zip(range(k), colors):
    k_members = sk_labels == i
    cluster_center = sk_centers[i]
    plt.plot(x[k_members, a], x[k_members, b], color + '.') #plot members
    plt.plot(cluster_center[a], cluster_center[b], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

#figure 3
c=0
d=2
fig3 = plt.figure(figsize=(9,7))
plt.subplot(2,1,1)
plt.xlabel(features[c])
plt.ylabel(features[d])
plt.title('Relations between '+features[c]+' vs '+features[d])
for i, color in zip(range(k), colors):
    k_members = learned_labels == i
    cluster_center = learned_centers[i]
    plt.plot(x[k_members, c], x[k_members, d], color + '.') #plot members
    plt.plot(cluster_center[c], cluster_center[d], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k
plt.subplot(2,1,2)
plt.xlabel(features[c])
plt.ylabel(features[d])
plt.title('Down is from sklearn. F1=%.4f' %f1, loc='right')
for i, color in zip(range(k), colors):
    k_members = sk_labels == i
    cluster_center = sk_centers[i]
    plt.plot(x[k_members, c], x[k_members, d], color + '.') #plot members
    plt.plot(cluster_center[c], cluster_center[d], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

plt.show()

data['label'] = learned_labels
f.create_csv('result', data)
#figures.append(fig2)
#figures.append(fig3)
#f.make_gif(figures, 'test_customers')