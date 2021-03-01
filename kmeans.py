import numpy as np
import random

class kmeans:

  def __init__(self, k, maxIterations=1000):
    self.k = k
    self.maxIterations = maxIterations

  def fit(self, X):
    self.X = X
    rows = X.shape[0]

    # Initialize starting centroids randomly
    self.centroids = X[np.random.permutation(rows)[:self.k]]
    
    for i in range(self.maxIterations):
      # Calculate the distance from each point to each cluster centroid
      distanceToCentroids = np.zeros((rows, self.k))
      for cluster in range(self.k):
        distanceToCentroid = np.sqrt(np.sum((X - self.centroids[cluster])**2, axis=1))
        distanceToCentroids[:, cluster] = distanceToCentroid
        
      # Assign points to clusters based on closest distance to a centroid
      self.assignedClusters = np.argmin(distanceToCentroids, axis=1)

      # Recalculate centroid locations based on the clusters' points

      previousCentroids = np.copy(self.centroids)
      for cluster in range(self.k):
        self.centroids[cluster] = np.mean(X[self.assignedClusters == cluster, :], axis=0)

      # If centroids didn't change, break out of the loop
      if np.all(previousCentroids == self.centroids):
        break

    return self.assignedClusters

  def predict(self, X):
    rows = X.shape[0]

    # Calculate the distance from each point to each clsuter centroid
    distanceToCentroids = np.zeros((rows, self.k))
    for cluster in range(self.k):
      distanceToCentroid = np.sqrt(np.sum((X - self.centroids[cluster])**2, axis=1))
      distanceToCentroids[:, cluster] = distanceToCentroid
        
      # Assign points to clusters based on closest distance to a centroid
      assignedClusters = np.argmin(distanceToCentroids, axis=1)

      return assignedClusters

  def generateData(self, n):
    cols = self.centroids.shape[1]
    X = np.zeros((n, cols))

    for i in range(n):
      # Determine a random cluster to generate an artificial data point from
      randomCluster = random.randrange(self.k)
      for j in range(cols):
        # Calculate min and max feature value for the given cluster
        min = np.amin(self.X[self.assignedClusters == randomCluster, j])
        max = np.amax(self.X[self.assignedClusters == randomCluster, j])
        # Generate new data as an average of two uniform random variables
        X[i, j] = (random.uniform(min, max) + random.uniform(min, max)) / 2

    return X

  def generateCategoricalData(self, n):
    cols = self.centroids.shape[1]
    X = np.zeros((n, cols))

    for i in range(n):
      # Determine a random cluster to generate an artificial data point from
      randomCluster = random.randrange(self.k)
      for j in range(cols):
        # Calculate unique feature values for the given cluster
        values = np.unique(self.X[self.assignedClusters == randomCluster, j])
        
        # Generate new data as a possible combination of categories
        values = np.random.permutation(values)
        X[i, j] = values[0]

    return X
      
    