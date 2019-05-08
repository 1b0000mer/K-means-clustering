import numpy as np
import matplotlib.pyplot as plt
from random import randint
from itertools import cycle

#function
def GenerateData(dim, N_data, N_groups, max_val, min_val):
    #initialization
    dataArray = np.ones((0,dim), dtype = int)
    true_lbl = []

    #genarate data of groups
    for i in range(N_groups):
        groupData = np.random.rand(N_data,dim) * (min_val[i] - max_val[i]) + max_val[i]

        #concatenated gen groups -> data
        dataArray = np.concatenate((dataArray,groupData))

        #genarate true label
        for j in range(N_data):
            true_lbl.append(i)
    return dataArray, true_lbl

#Euclidean norm
def cal(point1, point2):
    return  np.linalg.norm(point1 - point2)

#create base centers
def initCenters(max_val, min_val, dim, N_groups):
    #initialization
    centers = np.ones((0,dim), dtype = int)

    for i in range(N_groups):
        cal = np.random.rand(1,dim) * (min_val[i] - max_val[i]) + max_val[i]
        centers = np.concatenate((centers,cal))
    
    return centers;

#clustering
def kmeans(data, dim, N_groups, max_val=0, min_val=0, min_dist=0.01, ran_centers=True):
    '''
    ran_centers (optional):
        True (default) if you want centers create randomly
        False if you want centers are choose random from dataset
    '''

    #initialization

    base_centers = []
    if (ran_centers == True):
        base_centers = initCenters(max_val, min_val, dim, N_groups)
    else:
        for i in range(N_groups):
            base_centers = np.concatenate((base_centers, data[randint(0, len(data))]))
    flag = 0
    new_centers = []
    while (flag == 0):
        if (len(new_centers) == 0):
            old_centers = base_centers
        else:
            old_centers = new_centers
        new_centers = []
        cal_dist = []
        learned_lbl = []
        
        #calculate distance of data to centers
        for i in range(N_groups):
            dist_last = []
            for j in range(len(data)):
                dist = cal(old_centers[i], data[j])
                dist_last.append(dist)
            cal_dist.append(dist_last)

        learned_lbl = np.argmin(cal_dist, axis=0)

        #map data to new label and calculate new center 
        mydata_cluster = []
        for m in range(N_groups):
            temp = []
            for n in range(len(learned_lbl)):
                if (learned_lbl[n] == m):
                    temp.append(data[n])

            mydata_cluster.append(temp)
            new_centers.append(np.mean(temp, axis=0))

        #check if centers are stop moving
        for z in range(N_groups):
            if (cal(new_centers[z], old_centers[z]) < min_dist):
                flag = 1
            else:
                flag = 0
                
    return mydata_cluster, new_centers, learned_lbl

#F1 score
def F1_score(true_lbl, learned_lbl):
    #initialization
    TP=0; FN=0;FP=0; TN=0;

    for i in range(0, len(true_lbl)-1):
        for j in range(1, len(learned_lbl)):
            if (true_lbl[i] == true_lbl[j]):
                if (learned_lbl[i] == learned_lbl[j]):
                    TP=TP+1
                else:
                    FN=FN+1
            else:
                if (true_lbl[i] != true_lbl[j]):
                    if (learned_lbl[i] != learned_lbl[j]):
                        TN=TN+1
                    else:
                        FP=FP+1

    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*Precision*Recall/(Precision + Recall)

    return F1