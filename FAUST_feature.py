from plyfile import PlyData
import numpy as np
import networkx as nx
from scipy import sparse
from plyfile import PlyElement
from collections import defaultdict
import scipy.stats.mstats
from sklearn.svm import SVC
from collections import Counter


def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = np.transpose([np.array(data['vertex'][axis]) for axis in ['x', 'y', 'z']])

    faces = data['face']['vertex_indices']
    faces = [np.array(face) for face in faces]
    face = np.transpose(faces)

    return pos,face


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


def g(lam):
    return np.exp(-lam)

def h(lam,j):
    #lam is a numpy array
    return np.diag((g(lam*2**(j-1))**2 - g(lam*2**j)**2)**(1/2))



def calculate_wavelet(eigenval,eigenvec):
    dilation = [-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]
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


def zero_order_feature(Aj, M, shot):
    #this takes the shot features **shot** and low pass filter **Aj** in, and calculate the 0 order features
    #Besides, to do integration and convolution, we need to add the area matrix **M**, then sum it up finally to get 0 order features
    shot_weighted = M*shot
    F0 = np.sum(M*np.matmul(Aj,shot_weighted),0)
    return F0.reshape(-1,1)



def first_order_feature(psi,Wf,M,Aj,shot):
    #this takes in the shot features **shot**, area matrx **M**, the low pass filter **Aj**, and an array of precomputed psi*Aj*shot **Wf** 
    temp = np.einsum('i,aij -> aij',M.data,Wf)
    F1 = []
    for ele in temp:
        F1.append(np.matmul(Aj,ele))
    F1 = np.sum(np.einsum('i,aij -> aij',M.data,F1),1)
    #F = []
    #F.append(np.sum(np.matmul(P,u[0]),1))
    #for i in range(1,t):
    #F = np.concatenate((F,np.sum(np.matmul(P,u[i]))),1)
    return F1.reshape(-1,1)
    


def selected_second_order_feature(psi,Wf,M,Aj,shot):
    #this takes in the shot features **shot**, area matrx **M**, the low pass filter **Aj**, and an array of precomputed psi*Aj*shot **Wf**
    temp = np.einsum('i,aij -> aij',M.data,np.abs(Wf[0:1]))
    F2 = np.einsum('ij,ajt ->ait',psi[1],temp)
    for i in range(2,len(psi)):
        temp = np.einsum('i,aij -> aij',M.data,np.abs(Wf[0:i]))
        F2 = np.concatenate((F2,np.einsum('ij,ajt ->ait',psi[i],temp)),0)
    F2 = np.einsum('i,aij -> aij',M.data,np.abs(F2))
    F2 = np.einsum('ij,ajt -> ait',Aj,F2)
    temp = np.einsum('i,aij -> aij',M.data,F2)
    F2 = np.sum(temp,1)
    #F2.append(np.sum(np.einsum('ij,ajt ->ait',W[i],u[0:i]),1).reshape(len(u[0:i])*F,1))
    #F2 = np.sum(np.einsum('ijk,akt ->iajt',P,F1),2).reshape(len(P)*len(F1)*F,1)
    #F2 = np.array(F2).reshape()
    return F2.reshape(-1,1)
    


def generate_feature(psi,Wf,M,Aj,shot):
    #with zero order, first order and second order features
    #shall consider only zero and first order features
    F0 = zero_order_feature(Aj, M, shot)
    np.savetxt('MeshFeatureZero'+str(i)+'.txt',F0)
    F1 = first_order_feature(psi,Wf,M,Aj,shot)
    #F2 = second_order_feature(W,u,P[0],t,F)
    F2 = selected_second_order_feature(psi,Wf,M,Aj,shot)
    #F3 = selected_third_order_feature(W,u,P[0],t,F)
    F = np.concatenate((F0,F1),axis=0)
    np.savetxt('MeshFeatureFirst'+str(i)+'.txt',F)
    F = np.concatenate((F,F2),axis=0)
    np.savetxt('MeshFeature'+str(i)+'.txt',F)
    #F = np.concatenate((F1,F2),axis=0)
    #F = np.concatenate((F,F3),axis=0)
    return F


def read_shot(file):
    with open(file,'r') as f:
        data = f.readlines()
    feature = np.zeros((6890,352))
    for i in range(len(data)-1):
        line = data[i]
        vertex_index = int(line.split()[2][4:])
        feature[vertex_index] = np.array(line.split()[6:],dtype=np.float32)
    return feature


def read_label(i):
    return i%10

for i in range(100):
    if i <10:
        pos,face= read_ply('faust_file/tr_reg_00'+str(i)+'.ply')
    else:
        pos,face= read_ply('faust_file/tr_reg_0'+str(i)+'.ply')
    M = generate_area(pos,face)
    eigenval = np.loadtxt('/mnt/research/cedar_team/geometric_scattering/FAUST/eigenval/'+'eigenval'+str(i)+'.txt')
    eigenvec = np.loadtxt('/mnt/research/cedar_team/geometric_scattering/FAUST/eigenvec/'+'eigenvec'+str(i)+'.txt')
    eigenval[0] = 0
    norm_vec = eigenvec/np.sum(np.multiply(M*eigenvec,eigenvec),0)
    shot = read_shot('shot/'+'shot'+str(i)+'.txt')
    psi,Aj = calculate_wavelet(eigenval,norm_vec)
    Wf = weighted_wavelet_transform(psi,M,shot)
    feature = generate_feature(psi,Wf,M,Aj,shot)
