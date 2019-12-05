from plyfile import PlyData
import numpy as np
import networkx as nx
from scipy import sparse
from plyfile import PlyElement


def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = np.transpose([np.array(data['vertex'][axis]) for axis in ['x', 'y', 'z']])

    faces = data['face']['vertex_indices']
    faces = [np.array(face) for face in faces]
    face = np.transpose(faces)

    return pos,face

def face_to_edge(face):
    edge_pool = np.concatenate((face[:2],face[1:],face[::2]),1)
    edge_pool = np.transpose(edge_pool)
    return edge_pool



def create_graph(edge_pool):
    G = nx.Graph()
    G.add_edges_from(edge_pool)
    #G.add_weighted_edges_from
    return G

for t in range(100):
    if t<10:
        pos,face= read_ply('tr_reg_'+'00'+str(t)+'.ply')
    else:
        pos,face= read_ply('tr_reg_'+'0'+str(t)+'.ply')


    edge_pool = face_to_edge(face)
    graph = create_graph(edge_pool)


    I = []
    J = []
    V = []


    n = 6890
    for i in range(n):
        if i %100==0:
            print(i)
        neighbors = list(graph.neighbors(i))
        weights = []
        z = len(neighbors)
        I = I + ([i] * (z + 1)) # repeated row
        J = J + neighbors + [i] # column indices and this row
        for j in range(z):
            neighbor = neighbors[j]
            edge = nx.common_neighbors(graph,i,neighbor)
            #faces = [edge.f1, edge.f2]
            cotangents = []
            
            for node in edge:
                (u,v) = (pos[i] - pos[node], pos[neighbor] - pos[node])
                cotangents.append(np.dot(u, v) / np.sqrt(np.sum(np.square(np.cross(u, v)))))
            
            #for f in range(2):
            #    if faces[f]:
            #        P = mesh.VPos[filter(lambda v: v not in [neighbor, vertex], faces[f].getVertices())[0].ID]
            #        (u, v) = (mesh.VPos[vertex.ID] - P, mesh.VPos[neighbor.ID] - P)
            #        cotangents.append(np.dot(u, v) / np.sqrt(np.sum(np.square(np.cross(u, v)))))

            weights.append(1 / len(cotangents) * np.sum(cotangents)) # cotangent weights

        V = V + weights + [(-1 * np.sum(weights))] # n negative weights and row vertex sum



    L = sparse.coo_matrix((V, (I, J)), shape=(n, n)).tocsr()



    n = 6890


    #calculate areas of each faces
    from collections import defaultdict
    face_component = face.transpose()
    area_dict = {}
    node_dict = defaultdict(list)
    for f in face_component:
        area = np.linalg.norm(np.cross(pos[f[1]] - pos[f[0]],pos[f[2]] - pos[f[1]]))/2
        area_dict[tuple(f)] = area
        for node in f:
            node_dict[node].append(tuple(f))



    #finally the node area, eaquals the 1/3 of area of all adjacent triangels
    node_area = dict()
    node_A = []
    for i in range(n):
        temp_area = 0
        for f in node_dict[i]:
            temp_area += area_dict[f]
        node_area[i] = 1/3 * temp_area
        node_A.append(1/(1/3 * temp_area))


    Row = [i for i in range(n)]
    Col = [i for i in range(n)]
    M = sparse.coo_matrix((node_A, (Row, Col)), shape=(n,n)).tocsr()


    L_M = M*(-L)


    vals, vecs = sparse.linalg.eigs(L_M,k=512,which='SM')



    np.savetxt('eigenval'+str(t)+'.txt',vals.real)

    np.savetxt('eigenvec'+str(t)+'.txt',vecs.real)
