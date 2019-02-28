import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
from CNN_defines import *
from tensorflow.examples.tutorials.mnist import input_data

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

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Session
sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])
#first layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#second layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#third full connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool_flat2 = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat2, W_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#fourth full connected layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1_drop, W_fc2)+b_fc2
y = tf.nn.softmax(h_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h_fc2, labels = y_))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
batch_size = 100
#num_batch = dim[0]//batch_size
num_batch = 40000
init_op = tf.global_variables_initializer()
with sess.as_default():
    sess.run(init_op)
    for i in range(num_batch):
        start_index = get_random_index_from_data(train_y_, batch_size)
        batch_xt = train_x[start_index:(start_index+batch_size)][:]
        batch_yt = train_y_[start_index:(start_index+batch_size)][:]
        #batch = mnist.train.next_batch(batch_size)
        train_step.run({x:batch_xt, y_:batch_yt, keep_prob:0.5})
        #train_step.run({x:batch[0], y_:batch[1], keep_prob:0.5})
    test_y_pred = tf.argmax(y,1)
    y_pred = test_y_pred.eval({x:test_x, keep_prob:1.0})
    ImageId = [m+1 for m in range(y_pred.shape[0])]
    result = pd.DataFrame({
        'ImageId' : ImageId,
        'Label' : y_pred
        })
    result.to_csv(SCRIPT_PATH + "/submission_cnn_few_data.csv", index=False)
