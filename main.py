import numpy as np
import matplotlib.pyplot as plt
import sys
import kmeans

if len(sys.argv) < 3:
  print("Not enough arguments. Please execute main.py in the following format: 'py main.py <path_to_dataset> <continuous|discrete>'. Use continuous mode for continuous numeric data, and discrete mode for categorical data")
else:
  mode = sys.argv[2]

  # Read dataset filename from command line and load dataset as numpy array
  datasetFile = sys.argv[1]
  X = np.loadtxt(datasetFile, delimiter=",")

  # Separate dataset into features (X) and classes (Y)
  Y = X[:, 0]
  X = X[:, 1:]
  cols = X.shape[1]

  # Set number of clusters (k) to be the range of Y
  k = int(np.amax(Y) - np.amin(Y) + 1)

  # Apply k-means clustering to X
  model = kmeans.kmeans(k)
  Y_pred = model.fit(X)
  centroids = model.centroids

  # Generate some new data points
  newData = []
  goodMode = True
  if mode[0] == 'c':
    newData = model.generateData(10)
  elif mode[0] == 'd':
    newData = model.generateCategoricalData(10)
  else:
    goodMode = False
    print('mode not recognized. No new data will be generated.')

  # Determine the most cluster distinct features for plotting
  interclusterDistances = np.zeros((1, cols))

  for i in range(k):
    for j in range(k):
      if i == j:
        continue
      for feature in range(cols):
        interclusterDistances[0, feature] += abs(centroids[i, feature] - centroids[j, feature])

  feature0 = np.argmax(interclusterDistances)
  interclusterDistances[0, feature0] = 0
  feature1 = np.argmax(interclusterDistances)

  
  for i in range(k):
    plt.scatter(X[Y_pred == i, feature0], X[Y_pred == i, feature1], label=('cluster ' + str(i)))
    # Comment the above line and uncomment the below line to plot the dataset's actual categories rather than the determined clusters
    #plt.scatter(X[Y == i + np.amin(Y), feature0], X[Y == i + np.amin(Y), feature1], label=('category ' + str(i)))

  plt.scatter(centroids[:, feature0], centroids[:, feature1], marker='X', label='cluster centroid')
  
  if goodMode:
    plt.scatter(newData[:, feature0], newData[:, feature1], label='newly generated data')

  plt.xlabel('Feature ' + str(feature0))
  plt.ylabel('Feature ' + str(feature1))

  plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
  plt.show()