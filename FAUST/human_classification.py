import numpy as np
import networkx as nx
from scipy import sparse
from collections import defaultdict
import scipy.stats.mstats
from sklearn.svm import SVC
from collections import Counter

def read_label_people(i):
    return int(str(i)[0])
  
feature0 = []
feature1 = []
feature2 = []
Y = []
for i in range(100):
    temp = np.loadtxt('MeshFeatureFirst'+str(i)+'.txt')
    feature1.append(temp)
    Y.append(read_label_people(i))
for i in range(100):
    temp = np.loadtxt('MeshFeatureZero'+str(i)+'.txt')
    feature0.append(temp)
for i in range(100):
    temp = np.loadtxt('MeshFeature'+str(i)+'.txt')
    feature2.append(temp)
    
feature0 = np.reshape(feature0,(len(feature0),feature0[0].shape[0]))
feature1 = np.reshape(feature1,(len(feature1),feature1[0].shape[0]))
feature2 = np.reshape(feature2,(len(feature2),feature2[0].shape[0]))


feature_z0 = scipy.stats.mstats.zscore(feature0,0)
feature_z1 = scipy.stats.mstats.zscore(feature1,0)
feature_z2 = scipy.stats.mstats.zscore(feature2,0)


set1 = [i+j*10 for j in range(10) for i in [0,1]]
set2 = [i+j*10 for j in range(10) for i in [2,3]]
set3 = [i+j*10 for j in range(10) for i in [4,5]]
set4 = [i+j*10 for j in range(10) for i in [6,7]]
set5 = [i+j*10 for j in range(10) for i in [8,9]]


Y = np.array(Y)

Y_resample = np.concatenate((Y[set1],Y[set2],Y[set3],Y[set4],Y[set5]),0)

feature0_resample = feature_z0[(set1,set2,set3,set4,set5),:]
feature0_resample = np.concatenate(feature0_resample,0)

feature1_resample = feature_z1[(set1,set2,set3,set4,set5),:]
feature1_resample = np.concatenate(feature1_resample,0)

feature2_resample = feature_z2[(set1,set2,set3,set4,set5),:]
feature2_resample = np.concatenate(feature2_resample,0)




G_pool = [0.00001,0.0001]
C_pool = [25,500]


#this is for human classification
#index = np.arange(len(Y))
#np.random.shuffle(index)
#write_to_file(index)
n_splits = 5
#shuffled_feature0,shuffled_Y = shuffled(index,feature_z,Y)
train_id,test_id = Kfold(len(feature0_resample),n_splits)

#write_to_file(index)
#n_splits = 5
#train_id,test_id = Kfold(len(feature_z),n_splits)
outter_loop = 0
test_accuracy = []
for k in range(n_splits):
    
    print('begin cross validation')
    outter_loop = outter_loop +1
    print('outter loop',outter_loop,'start')
    train_all_feature = []
    test_feature = []
    
    train_all_feature0 = [feature0_resample[i] for i in train_id[k]]
    train_all_feature1 = [feature1_resample[i] for i in train_id[k]]
    train_all_feature2 = [feature2_resample[i] for i in train_id[k]]
    
    train_all_Y = [Y_resample[i] for i in train_id[k]]
    
    train_all_feature.append(train_all_feature0)
    train_all_feature.append(train_all_feature1)
    train_all_feature.append(train_all_feature2)
    
    test_feature0 = [feature0_resample[i] for i in test_id[k]]
    test_feature1 = [feature1_resample[i] for i in test_id[k]]
    test_feature2 = [feature2_resample[i] for i in test_id[k]]
    
    test_feature.append(test_feature0)
    test_feature.append(test_feature1)
    test_feature.append(test_feature2)

    test_Y = [Y_resample[i] for i in test_id[k]]
    
    result,prediction_acc = cross_validate(8,train_all_feature,train_all_Y,test_feature,test_Y,outter_loop,G_pool,C_pool)
    #run_train(session, train_all_feature, train_all_Y)
    print("Cross-validation result: %s" % result)
    print('prediction accuracy',prediction_acc)
    test_accuracy.append(prediction_acc)
    print('outter loop',outter_loop,'ends')
#print("Test accuracy: %f" % session.run(accuracy, feed_dict={fea: test_feature, clas: test_Y}))
print(test_accuracy)
print('mean accuracy is ', np.mean(test_accuracy))
print('std is', np.std(test_accuracy))
