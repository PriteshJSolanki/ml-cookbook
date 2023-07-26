'''
K-Means Clustering

K Means Clustering is an unsupervised learning algorithm (i.e it take unlabeled data) 
that will attempt to group similar clusters together.

The algorithm performs the following steps:
1. Choose a number of Clusters “K”
2. Randomly assign each point to a cluster 
3. For each cluster, compute the centroid by taking the mean vector of points in the cluster
4. Assign each data point to the cluster for which the centroid is the closest
5. Repeat until the clusters stop changing

Selecting K:
There are many methods, but a common oway is to use the elbow method. 
1. Compute a sum of squared error (SSE) for some value K. Increase K until the SSE drops abruptly.

This script will use create a random set of blobs and use the k-means clustering algorithm to 
assign them various labels

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class KMC:
    def __init__(self) -> None:
        self.data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)
        self.model = None
        self.X_test = None
        self.y_test = None

    def eda(self):
        """
        Exploratory Data Analysis

        """
        plt.scatter(self.data[0][:,0],self.data[0][:,1],c=self.data[1],cmap='rainbow')
        plt.show()

    def train(self, k:int=None):
        """
        Train the model

        """
        # Fit the model to all the features except Private
        self.model = KMeans(n_clusters=4) 
        self.model.fit(self.data[0])
        print(self.model.cluster_centers_)
        print(self.model.labels_)

    def eval(self):
        """
        Evaluate the model

        """
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,6))
        ax1.set_title('K Means')
        ax1.scatter(self.data[0][:,0],self.data[0][:,1],c=self.model.labels_,cmap='rainbow')
        ax2.set_title("Original")
        ax2.scatter(self.data[0][:,0],self.data[0][:,1],c=self.data[1],cmap='rainbow')
        plt.show()
        
if __name__ == '__main__':
    kmc = KMC()
    kmc.eda()
    kmc.train()
    kmc.eval()
