# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from collections import Counter
import pandas as pd
import random

# load dataset
print("Loading data...")
df = pd.read_csv('breast-cancer-wisconsin.data.txt')    #import raw dataset
df.replace('?', -99999, inplace=True)   #repair missing data to id outliers, inplace=T --> do right away
df.drop(['id'], 1, inplace=True)    #drop the id column otherwise our accuracy is relegated to coinflip status
full_data = df.astype(float).values.tolist()

#shuffle the data
print("Shuffling data...")
random.shuffle(full_data)

#slicing the data into train/test sets
print("Slicing data...")
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size * len(full_data))] #first 80%
test_data = full_data[-int(test_size * len(full_data)):]  #last 20%

for i in train_data:
    # append the sets in the list up to the class
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    # append the sets in the list up to the class
    test_set[i[-1]].append(i[:-1])

#################

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
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print('Confidence of Incorrect Votes:', confidence)
        total += 1
print('Accuracy:', correct/total)
