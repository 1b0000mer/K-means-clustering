import matplotlib.pyplot as plt
from itertools import cycle
import function as f
import csv

#initialization for generate samples data
dim = 2; 'feature'
N_data = 50
N_groups = 6
max_val = [2, 1, 2, 3, 2, 3]; 'must same as group'
min_val = [1, 2, 2, 4, 1, 4];

#main: create samples data
data = f.GenerateData(dim, N_data, N_groups, max_val, min_val)
kq = f.kmeans(data[0], dim, N_groups, max_val, min_val)
f1 = f.F1_score(data[1], kq[2])

print('Number of data: %d' % len(data[0]))
print('Number of clusters: %d' % len(kq[0]))
print('F1 Score: %.4f' % f1)

#plot data
plt.figure(1)
plt.title('Original')
j=0
for i in range(N_groups*N_data):
    plt.plot(data[0][i][j], data[0][i][j+1], '.')
    j=0

plt.figure(2)
#plot data
j=0
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for i, col in zip(range(len(kq[0])), colors):
    for q in range(len(kq[0][0])):
        plt.plot(kq[0][i][q][j], kq[0][i][q][j+1], col + '.')
    j=0

#plot centers
n=0
for m, col in zip(range(N_groups), colors):
    plt.plot(kq[1][m][n], kq[1][m][n+1], col + '^', markersize=14)
    n=0

plt.title('Number of data: %d. Number of clusters: %d. F1 Score: %.4f' % (len(data[0]), len(kq[0]), f1))
plt.show()

##main: read data from csv file
#data_origin=[]
#refactor_data=[]
#label=[]
#N_data=-1
#N_groups=3
#dim=0

#data_file_name='dataset_for_clustering.csv'
#d=','

#c = open(data_file_name, 'r')

#csv_reader = csv.reader(c, delimiter=d)
#dim=len(next(csv_reader)) - 1 # Read first line and count columns
#c.seek(0) # go back to beginning of file
#for row in csv_reader:
#    data_origin.append(row)
#    N_data += 1

#refactor_data = data_origin

#refactor_data.pop(0)
#for i in range(len(refactor_data)):
#    del refactor_data[i][0]

#kq = f.kmeans(refactor_data,dim,N_groups,ran_centers=False)