# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

        #class k contains features [[...]]
dataset = {'b': [ [1,2], [2,3], [3,1] ], 'r': [ [6,5], [7,7], [8,6] ] }
new_features = [5,7]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('invalid k value --> less than #voting groups')
    #upgrade later to utilize radius for optimization
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm( np.array(features) - np.array(predict) ) #synonymous with long euclid distance formula
            distances.append( [euclidean_distance, group] ) #add the distance and its group to the list

    votes = [ i[1] for i in sorted(distances)[:k] ] #rank them according to distance
    vote_result = Counter(votes).most_common(1)[0][0] # snag the most commonly voted
    return vote_result

result = k_nearest_neighbors(dataset, new_features)
print(result)

#show the dataset
for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()
