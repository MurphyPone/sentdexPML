# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

# load dataset
df = pd.read_csv('breast-cancer-wisconsin.data.txt')    #import raw dataset
df.replace('?', -99999, inplace=True)   #repair missing data to id outliers, inplace=T --> do right away
df.drop(['id'], 1, inplace=True)    #drop the id column otherwise our accuracy is relegated to coinflip status

# define Feature (Xs) and Labels/Class(ys)
X = np.array(df.drop(['class'],1)) # everything but the class
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) # organize data into training/testing clumps

# configure the model
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# predict!
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
