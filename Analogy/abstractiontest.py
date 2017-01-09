from pprint import pprint
from utils import AIMind
from collections import Counter
import numpy as np
import statistics
from scipy.spatial.ckdtree import cKDTree


#d1 = AIMind("data files/roman_empire_1000.xml").as_domain()
d1 = AIMind("data files/ww2.xml").as_domain()
d2 = AIMind("data files/roman_empire_1000.xml").as_domain()

u1 = {(r,):{src for (src,dst) in s} for r,s in d1.usage_map.items()}
u2 = {(r,):{src for (src,dst) in s} for r,s in d2.usage_map.items()}



def normalize(c):
    total = sum(c.values())
    return {x:c[x]/total for x in c}


def merge(u,c):
    u2 = {}
    for r1,s1 in u.items():
        for r2,s2 in u.items():
            key = tuple(sorted(list(set(r1)|set(r2))))
            if r1 != r2 and key not in u2:
                v = s1&s2
                if len(v) != 0:
                    c[key] = len(v)
                    u2[key] = v
    return u2


def filter(u,c):
    #threshold: >1 standard deviations from the mean
    x = [v for k,v in normalize(c).items()]
    t = statistics.mean(x) + 1*statistics.stdev(x)
    n = normalize(c)
    return {k:v for k,v in u.items() if n[k] > t}

def abstract(u):
    c = Counter()
    u1 = u.copy()
    u2 = u1.copy()
    while True:
        l = len(u1)
        print(l)
        u3 = merge(u2,c)
        if len(u3) < 2:
            u1.update(u3)
            break
        uf = filter(u3,c)
        u1.update(uf)
        u2 = uf
        c.clear()
    return u1
   
def convert(d,u):
    u1 = {}
    for c,k in u.items():
        v = np.mean([d.rtype_vectors[x] for x in c],axis=0)
        for j in k:
            u1.setdefault(j,[]).append(v)
    return u1
            

uf1 = abstract(u1)
uf2 = abstract(u2)

x1 = convert(d1,uf1)
x2 = convert(d2,uf2)

kd1_keys = []
kd1_values = []
for k,v in x1.items():
    for c in v:
        kd1_keys.append(k)
        kd1_values.append(c)

kd2_keys = []
kd2_values = []
for k,v in x2.items():
    for c in v:
        kd2_keys.append(k)
        kd2_values.append(c)

kd1 = cKDTree(kd1_values)
kd2 = cKDTree(kd2_values)


def query(kdtree,kdkeys,vals,n=1):
    tmp = zip(*kdtree.query(vals,n))
    return sorted([(d, kdkeys[i]) for d,i in tmp])

pprint(query(kd2,kd2_keys,x1["Adolf Hitler"]))