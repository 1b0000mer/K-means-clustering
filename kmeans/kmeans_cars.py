import matplotlib.pyplot as plt
from itertools import cycle
import function as f
from sklearn.datasets import make_blobs
from itertools import cycle
import numpy as np
import pandas as pd
import os

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

file_path = os.getcwd() + '\data\cars\dataset_for_clustering.csv'
features = ['vehicle_name', 'fuel_consumption', 'num_cylinders', 'engine_volumetric', 'horse_power', 'rear_ratio', 'vehicle_weight', 'vehicle_acceleration', 'engine_cylinder_type', 'transmission_type', 'num_gears', 'num_carburetors']

data = f.read_file(file_path, features, 'vehicle_name', 'vehicle_name')
data, features = f.remove_features(data, features, ['num_cylinders','engine_cylinder_type', 'transmission_type', 'num_gears', 'num_carburetors'])
features.remove('vehicle_name') #index_col does not count
x = np.array(data.values, dtype=float)
n_features = len(features)
k = 3

#plot data
fig1 = plt.figure(1)
fig1 = plt.axes(projection='3d')
plt.title('Original')
fig1.set_xlabel('fuel_consumption')
fig1.set_ylabel('engine_volumetric')
fig1.set_zlabel('horse_power')
fig1.scatter3D(x[:,features.index('fuel_consumption')], x[:,features.index('engine_volumetric')], x[:,features.index('horse_power')])

##kmeans from sklearn
from sklearn.cluster import KMeans
clustering = KMeans(k, random_state=None).fit(x)
sk_labels = clustering.labels_
sk_centers = clustering.cluster_centers_

while True:
    result, learned_centers, learned_labels, figures = f.new_kmeans(x, n_features, k, rand_centers=False, gif=False)
    f1 = f.F1_score(sk_labels, learned_labels)
    if (0.7 <= f1 <= 1):
        break
    print ('\nre run, F1=%.4f' %f1, end='\n')

print('F1 score=%.4f' %f1)

fig2 = plt.figure(figsize=(9,7))
plt.subplot(2,1,1)
plt.xlabel(features[0])
plt.ylabel(features[2])
plt.title('Relations between '+features[0]+' vs '+features[2])
for i, color in zip(range(k), colors):
    k_members = learned_labels == i
    cluster_center = learned_centers[i]
    plt.plot(x[k_members, 0], x[k_members, 2], color + '.') #plot members
    plt.plot(cluster_center[0], cluster_center[2], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k
plt.subplot(2,1,2)
plt.xlabel(features[0])
plt.ylabel(features[2])
plt.title('Down is from sklearn. F1=%.4f' %f1, loc='right')
for i, color in zip(range(k), colors):
    k_members = sk_labels == i
    cluster_center = sk_centers[i]
    plt.plot(x[k_members, 0], x[k_members, 2], color + '.') #plot members
    plt.plot(cluster_center[0], cluster_center[2], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

fig3 = plt.figure(figsize=(9,7))
plt.subplot(2,1,1)
plt.xlabel(features[1])
plt.ylabel(features[3])
plt.title('Relations between '+features[1]+' vs '+features[3])
for i, color in zip(range(k), colors):
    k_members = learned_labels == i
    cluster_center = learned_centers[i]
    plt.plot(x[k_members, 1], x[k_members, 3], color + '.') #plot members
    plt.plot(cluster_center[1], cluster_center[3], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k
plt.subplot(2,1,2)
plt.xlabel(features[1])
plt.ylabel(features[3])
plt.title('Down is from sklearn. F1=%.4f' %f1, loc='right')
for i, color in zip(range(k), colors):
    k_members = sk_labels == i
    cluster_center = sk_centers[i]
    plt.plot(x[k_members, 1], x[k_members, 3], color + '.') #plot members
    plt.plot(cluster_center[1], cluster_center[3], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

plt.show()

data['label'] = learned_labels
f.create_csv('result', data)
