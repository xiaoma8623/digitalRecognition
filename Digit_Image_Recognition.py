import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv

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
sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
Z3 = tf.matmul(x,W)+b
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
batch_size = 100
num_batch = dim[0]//batch_size
init_op = tf.global_variables_initializer()
with sess.as_default():
    sess.run(init_op)
    for i in range(num_batch):
        batch_xt = train_x[i*batch_size:((i+1)*batch_size-1)][:]
        batch_yt = train_y_[i*batch_size:((i+1)*batch_size-1)][:]
        train_step.run({x:batch_xt, y_:batch_yt})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #tess = tf.argmax(y,1)
    #print(tess.eval({x:batch_xt}))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(accuracy.eval({x:train_x, y_:train_y_}))
    test_y_pred = tf.argmax(y,1)
    y_pred = test_y_pred.eval({x:test_x})
    ImageId = [m+1 for m in range(y_pred.shape[0])]
    result = pd.DataFrame({
        'ImageId' : ImageId,
        'Label' : y_pred
        })
    result.to_csv(SCRIPT_PATH + "/submission.csv", index=False)
    
