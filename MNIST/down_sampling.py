import numpy as np
import pickle
from collections import defaultdict
import scipy.stats.mstats
from scipy import sparse

p = pickle.load(open('Downloads/icosphere_3.pkl', "rb"))
V = p['V']


import math, random

def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points
    
    
node_pool = fibonacci_sphere(4,randomize=False)

r = []
for j in node_pool:
    near = 10
    temp = 1000
    for i,pos in enumerate(V):
        if np.linalg.norm(pos-j) < near:
            temp = i
            near = np.linalg.norm(pos-j)
    r.append(temp)
    
selected = []
for j in range(46):
    for i in r:
        selected.append(i+j*642)


np.save('selected_node0',selected)
