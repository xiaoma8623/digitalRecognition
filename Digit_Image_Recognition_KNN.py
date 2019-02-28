import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
from sklearn.neighbors import KNeighborsClassifier

#get training data
#with open('train.csv') as csvfile:
#    readCSV = csv.reader(csvfile)
#    for row in readCSV:
#        print(row)
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
train_df = pd.read_csv(SCRIPT_PATH + "/train.csv")
dim = train_df.shape
train_y = train_df.iloc[:,0]
train_y = train_y.as_matrix()
train_y_ = np.zeros((train_y.shape[0],10))
train_y_[np.arange(train_y.shape[0]),train_y] = 1
train_x = train_df.iloc[:,1:]
train_x = train_x.as_matrix()

test_df = pd.read_csv(SCRIPT_PATH + "/test.csv")
test_x = test_df.as_matrix()

#Session
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_x, train_y)
Y_pred = knn.predict(test_df)
ImageId = [m+1 for m in range(Y_pred.shape[0])]
result = pd.DataFrame({
        'ImageId' : ImageId,
        'Label' : Y_pred
        })
result.to_csv(SCRIPT_PATH + "/submission_knn.csv", index=False)