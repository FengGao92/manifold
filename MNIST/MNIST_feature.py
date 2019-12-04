import numpy as np
import pickle
from collections import defaultdict
import scipy.stats.mstats
from scipy import sparse



def g(lam):
    return np.exp(-lam)

def h(lam,j):
    #lam is a numpy array
    return np.diag((g(lam*2**(j-1))**2 - g(lam*2**j)**2)**(1/2))



def calculate_wavelet(eigenval,eigenvec,J):
    dilation = np.arange(-8,J+1).tolist()
    wavelet = []
    for dil in dilation:
        wavelet.append(np.matmul(np.matmul(eigenvec,h(eigenval,dil)),np.transpose(eigenvec)))
    return wavelet,np.matmul(np.matmul(eigenvec,np.diag(g(eigenval*2**dilation[-1]))),np.transpose(eigenvec))



def weighted_wavelet_transform(wavelet,M,shot):
    #wavelet is the precomputed wavelet array, M is area matrix, shot is featrue matrix
    #calculates W*A*shot which approximates the convolution of W*shot
    Wf = []
    w_shot = M*shot
    #w_W=np.einsum('ijk,kt ->ijt',wavelet,w_shot)
    #needs to see which one is faster
    for psi in wavelet:
        Wf.append(np.matmul(psi,w_shot))
    return Wf



def generate_area(pos,face):
    face_component = face.transpose()
    area_dict = {}
    node_dict = defaultdict(list)
    for f in face_component:
        area = np.linalg.norm(np.cross(pos[f[1]] - pos[f[0]],pos[f[2]] - pos[f[1]]))/2
        area_dict[tuple(f)] = area
        for node in f:
            node_dict[node].append(tuple(f))
            
    node_area = dict()
    node_A = []
    n = pos.shape[0]
    for i in range(n):
        temp_area = 0
        for f in node_dict[i]:
            temp_area += area_dict[f]
        node_area[i] = 1/3 * temp_area
        node_A.append(1/3 * temp_area)


    Row = [i for i in range(n)]
    Col = [i for i in range(n)]
    M = sparse.coo_matrix((node_A, (Row, Col)), shape=(n,n)).tocsr()
    return M



p = pickle.load(open('icosphere_3.pkl', "rb"))
V = p['V']
F = p['F']



def zero_order_feature(Aj, M, shot):
    #this takes the shot features **shot** and low pass filter **Aj** in, and calculate the 0 order features
    #Besides, to do integration and convolution, we need to add the area matrix **M**, then sum it up finally to get 0 order features
    shot_weighted = M*shot
    F0 = np.matmul(Aj,shot_weighted)
    return F0.reshape(-1,1)



def first_order_feature(psi,Wf,M,Aj,shot):
    #this takes in the shot features **shot**, area matrx **M**, the low pass filter **Aj**, and an array of precomputed psi*Aj*shot **Wf** 
    temp = np.einsum('i,ai -> ai',M.data,np.abs(Wf))
    F1 = []
    for ele in temp:
        F1.append(np.matmul(Aj,ele))
    
    #F = []
    #F.append(np.sum(np.matmul(P,u[0]),1))
    #for i in range(1,t):
    #F = np.concatenate((F,np.sum(np.matmul(P,u[i]))),1)
    return np.reshape(F1,(-1,1))
    


def selected_second_order_feature(psi,Wf,M,Aj,shot):
    #this takes in the shot features **shot**, area matrx **M**, the low pass filter **Aj**, and an array of precomputed psi*Aj*shot **Wf** 
    temp = np.einsum('i,ai -> ai',M.data,np.abs(Wf[0:1]))
    F2 = np.einsum('ij,aj ->ai',psi[1],temp)
    for i in range(2,len(psi)):
        temp = np.einsum('i,ai -> ai',M.data,np.abs(Wf[0:i]))
        F2 = np.concatenate((F2,np.einsum('ij,aj ->ai',psi[i],temp)),0)
    F2 = np.einsum('i,ai -> ai',M.data,np.abs(F2))
    F2 = np.einsum('ij,aj -> ai',Aj,F2)
    #F2.append(np.sum(np.einsum('ij,ajt ->ait',W[i],u[0:i]),1).reshape(len(u[0:i])*F,1))
    #F2 = np.sum(np.einsum('ijk,akt ->iajt',P,F1),2).reshape(len(P)*len(F1)*F,1)
    #F2 = np.array(F2).reshape()
    return F2.reshape(-1,1)
    


def generate_feature(psi,Wf,M,Aj,shot):
    #with zero order, first order and second order features
    #shall consider only zero and first order features
    F0 = zero_order_feature(Aj, M, shot)
    F1 = first_order_feature(psi,Wf,M,Aj,shot)
    #F2 = second_order_feature(W,u,P[0],t,F)
    F2 = selected_second_order_feature(psi,Wf,M,Aj,shot)
    #F3 = selected_third_order_feature(W,u,P[0],t,F)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)
    #F = np.concatenate((F1,F2),axis=0)
    #F = np.concatenate((F,F3),axis=0)
    return F


#calculate area, read in eigenvec and eigenval
M = generate_area(V,F)
eigenval = np.loadtxt('eigenval.txt')
eigenvec = np.loadtxt('eigenvec.txt')
training_signal = np.load('MNIST_train.npy')
test_signal = np.load('MNIST_test.npy')
training_Y = np.load('MNIST_train_label.npy')
test_Y = np.load('MNIST_test_label.npy')
train_indices = [i for i, x in enumerate(training_Y) if x == 6]
test_indices = [i for i, x in enumerate(test_Y) if x == 6]
training_signal_ = [x for i, x in enumerate(training_signal) if i not in train_indices]
test_signal_ = [x for i, x in enumerate(test_signal) if i not in test_indices]
training_Y_ = [x for i, x in enumerate(training_Y) if i not in train_indices]
test_Y_ = [x for i, x in enumerate(test_Y) if i not in test_indices]


training_feature = []
for i in range(54082):
    if i %100==0:
        print(i)
    psi,Aj = calculate_wavelet(eigenval,eigenvec,0)
    Wf = weighted_wavelet_transform(psi,M,training_signal_[i])
    training_feature.append(generate_feature(psi,Wf,M,Aj,training_signal_[i]))
    
np.save('train_feature_J0'+'_'+str(i),training_feature)


training_feature = np.reshape(training_feature,(len(training_feature),training_feature[0].shape[0]))
np.save('train
test_feature = []
for i in range(9042):
    if i %100==0:
        print(i)
    psi,Aj = calculate_wavelet(eigenval,eigenvec,0)
    Wf = weighted_wavelet_transform(psi,M,test_signal_[i])
    test_feature.append(generate_feature(psi,Wf,M,Aj,test_signal_[i]))

test_feature = np.reshape(test_feature,(len(test_feature),test_feature[0].shape[0]))
np.save('test_featureJ0',test_feature)

