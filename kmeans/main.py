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

fig2 = plt.figure(2, figsize=(12,6))
plt.title('Data with true labels')
for i, color in zip(range(k), colors):
    k_members = labels_true == i
    cluster_center = centers[i]
    plt.plot(data[k_members, 0], data[k_members, 1], color + '.') #plot members
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=color,
             markeredgecolor='k', markersize=14) #plot k

plt.show()

result, new_centers, labels_learned, figures = f.new_kmeans(data, dim, k, rand_centers = False, gif=False)
f1 = f.F1_score(labels_true, labels_learned)


fig3 = plt.figure(3)
for i, color in zip(range(k), colors):
    temp = np.array(result[i])
    plt.scatter(temp[:,0], temp[:,1])

fig4 = plt.figure(4, figsize=(12,6))
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
#f.make_gif(figures, 'test4') #all figures must be same size!

'''real_data
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
'''

'''real_data_2
file_path = os.getcwd() + '\data\customers\Mall_Customers_edited.csv'
features = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

data = f.read_file(file_path, features, 'CustomerID')
features.remove('CustomerID') #index_col does not count
x = np.array(data.values, dtype=float)
n_features = len(features)
k = 3

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
a=1
b=2
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
c=1
d=3
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
'''