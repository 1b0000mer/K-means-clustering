import numpy as np
import matplotlib.pyplot as plt
from random import randint
from itertools import cycle
from io import BytesIO
import imageio
import time
import pandas as pd

def _cal(x):
    '''Euclidean norm'''
    return np.linalg.norm(x)

def _new_init_centers(features, data):
    '''create base center by random from range(min(data), max(data))'''
    min_value = np.min(data)
    max_value = np.max(data)
    center = np.random.randint((min_value+max_value)/2, max_value, size=(1,features))
    return center

def _init_centers(y_val, x_val, dim, N_groups):
    #initialization
    centers = np.ones((0,dim), dtype = int)

    for i in range(N_groups):
        cal = np.random.rand(1,dim) * (x_val[i] - y_val[i]) + y_val[i]
        centers = np.concatenate((centers,cal))
    
    return centers;

def _plot_iteration(data, labels, centers, iter=None, x_col=None, y_col=None, show=False, real_data=False):
    a=0
    b=1

    if real_data:
        a=1
        b=3

    fig = plt.figure(figsize=(9,7))
   
    if (iter != None):
        plt.title('iteration: %d' % iter)
    else:
        plt.title('relations between ' +x_col +' and ' +y_col)
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    colors = cycle('bgrcmyk')
    for i, color in zip(range(len(centers)), colors):
        k_members = labels == i
        cluster_center = centers[i]
        plt.plot(data[k_members, a], data[k_members, b], color + '.')
        plt.plot(cluster_center[a], cluster_center[b], '^', markerfacecolor=color, markeredgecolor='k', markersize=14)

    if show:
        plt.show()
    return fig

##function
def gen_data(dim, N_data, N_groups, y_val, x_val):
    #initialization
    dataArray = np.ones((0,dim), dtype = int)
    true_lbl = []

    #genarate data of groups
    for i in range(N_groups):
        groupData = np.random.rand(N_data,dim) * (x_val[i] - y_val[i]) + y_val[i]

        #concatenated gen groups -> data
        dataArray = np.concatenate((dataArray,groupData))

        #genarate true label
        for j in range(N_data):
            true_lbl.append(i)
    return dataArray, true_lbl

def new_kmeans(data, features, k, threshold=0.001, rand_centers=True, centers=[], gif=True):
    '''
    clustering data using kmeans
    +features: dimension of the data (columns)
    +threshold: acceptable threshold between new centers with old centers (in case kmeans never convergence)
    +ran_centers (optional):
        True (default) if you want centers create randomly
        False if you want centers are choose random from dataset
    +gif (optional):
        True (default) if you want return all figures for create gif file
    '''
    #data = np.array(data, dtype=float)
    print('K-means begins')
    start_time = time.time()
    
    size = (k,features)
    base_centers = np.zeros(shape=size) #(row,col)
    if (len(centers) > 0):
        base_centers = np.array(centers)
        print('choosen centers: \n', base_centers)
    else:
        if (rand_centers):
            for i in range(k):
                base_centers[i] = _new_init_centers(features, data)
            print('Done init centers! Choosen centers: \n', base_centers)
        else:
            for i in range(k):
                base_centers[i] = data[randint(0, len(data)-1)]
            print('choosen centers: \n', base_centers)

    old_centers = np.array(base_centers)
    new_centers = np.array(base_centers)
    learned_label = np.zeros(shape=(len(data), 1))
    distance_ = np.zeros(shape=(k,len(data)))
    iter = 0
    flag = False
    figures = None
    if gif:
        figures = []
    
    while (flag == False):
        #calculate distance of data to each center
        for i in range(k):
            for j in range(len(data)):
                distance_[i,j] = _cal(np.subtract(old_centers[i], data[j]))

        learned_label = np.argmin(distance_, axis=0)

        #map data to cluster and calculate new center 
        mydata_clusters = []
        for i in range(k):
            k_members = []
            for j in range(len(learned_label)):
                if (learned_label[j] == i):
                    k_members.append(data[j])

            mydata_clusters.append(k_members)
            if (len(k_members) > 0):
                new_centers[i] = np.mean(k_members, axis=0)

        if gif:
            fig = _plot_iteration(data, learned_label, new_centers, iter, real_data=False)
            figures.append(fig)
            plt.close()

        iter += 1
        print(iter," iteration(s) completed", end='\r')

        #check if centers are stop moving or meet threshold
        if (np.allclose(new_centers, old_centers) or _cal(np.subtract(new_centers, old_centers)) < threshold or iter==500):
            flag = True
            print('\ncenters: \n', new_centers)
            time_taken = time.time() - start_time
            print('time taken: %.4f (s)' % time_taken)

        else:
            flag = False
            old_centers = np.array(new_centers)
            new_centers = np.zeros(shape=size)
    
    return mydata_clusters, new_centers, learned_label, figures

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
    print('Precision: %.4f' %Precision)
    Recall = TP/(TP+FN)
    print('Recall: %.4f' %Recall)
    F1 = 2*Precision*Recall/(Precision + Recall)

    return F1

def make_gif(figures, file_name, **kwargs):
    '''
    Make gif result
    figures: fig array
    filename
    fps: default is 10
    '''
    print('Creating gif')
    try:
        if (1 <= len(figures) <= 10):
            fps = 1
        elif (11 <= len(figures) <= 20):
            fps = 2
        else:
            fps = 5
 
        images = []
        output = BytesIO()
        figures[0].savefig(output) 
        plt.close()
        output.seek(0)
        images.append(imageio.imread(output))
        images.append(imageio.imread(output))
        for fig in figures:
            output = BytesIO()
            fig.savefig(output)
            plt.close(fig)  
            output.seek(0)
            images.append(imageio.imread(output))

        images.append(imageio.imread(output))
        images.append(imageio.imread(output))

        filename = file_name+'.gif'
        imageio.mimsave(filename, images, fps=fps, **kwargs)
        print(filename, ' created successfully!')
    except:
        print('Can not create gif file!')

def read_file(FILE_PATH, FEATURE, INDEX_COL=None, HEADER=None):
    try:
        data = pd.read_csv(FILE_PATH, names=FEATURE, index_col=INDEX_COL)

        if (HEADER != None):
            data = data.drop(HEADER)
        print('Read csv file successfully')
        print(data)
        print('-------------------------------------------------------------------')
        return data
    except:
        print('Can not read csv file!')

def remove_features(data, features, arr_features):
    try:
        data = data.drop(columns=arr_features)
        for i in range(len(arr_features)):
            features.remove(arr_features[i])

        return data, features
    except:
        print('Failed!')

def create_csv(file_name, data):
    '''
    data: DataFrame
    using pandas
    '''
    print('Creating result.csv')
    try:
        data2 = data.sort_values(by='label', ascending=True)
        #print(data2.head())
        filename = file_name+'.csv'
        data2.to_csv(filename)
        print(filename, ' created successfully!')

    except:
        print('Can not create csv file!')