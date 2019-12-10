import numpy as np
import pickle
from collections import defaultdict
import scipy.stats.mstats
from sklearn.svm import SVC
from scipy import sparse
from collections import Counter
from utility import *

#read in selected nodes for j = 0 to perform downsampling.
selected_feature = np.load('selected_node0.npy')

training_Y = np.load('MNIST_train_label.npy')
test_Y = np.load('MNIST_test_label.npy')



train_indices = [i for i, x in enumerate(training_Y) if x == 6]
test_indices = [i for i, x in enumerate(test_Y) if x == 6]



training_signal = np.load('train_featureJ0.npy')

test_signal = np.load('test_featureJ0.npy')


training_Y_ = [x for i, x in enumerate(training_Y) if i not in train_indices]
test_Y_ = [x for i, x in enumerate(test_Y) if i not in test_indices]

#as an example
G_pool = [0.04]
C_pool = [650]

training_feature = np.reshape(training_signal,(len(training_signal),training_signal[0].shape[0]))
training_feature = training_feature[:,selected_feature]
#the first time save downsampling features
np.save('train_downsample',training_feature)
#once saved these can be used in the future
#training_feature = np.load('train_downsample.npy')
#training_feature_z = scipy.stats.mstats.zscore(training_feature,0)



test_feature = np.reshape(test_signal,(len(test_signal),test_signal[0].shape[0]))
test_feature = test_feature[:,selected_feature]
np.save('test_downsample',test_feature)
#test_feature = np.load('test_downsample.npy')
#test_feature_z = scipy.stats.mstats.zscore(test_feature,0)
print('begin cross validation')

result,prediction_acc = cross_validate(5,training_feature_z,training_Y_,test_feature_z,test_Y_,G_pool,C_pool)
#run_train(session, train_all_feature, train_all_Y)
print("Cross-validation result: %s" % result)
print('prediction accuracy',prediction_acc)
#print("Test accuracy: %f" % session.run(accuracy, feed_dict={fea: test_feature, clas: test_Y}))
