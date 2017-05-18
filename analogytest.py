from analogy.core import *
from analogy.utils import *
from pprint import pprint
import random

import time

from cProfile import run

import time


import pickle

import numpy as np

def process_analogy(x):
    pprint(x)
    print(explain_analogy(x))
    pprint(explain_analogy(x,paragraph=True))

def fba(x,d1,d2,cluster_mode=0,rmax=1,vmax=1,filter_list=None,knn_filter=None):
    start = time.time()
    process_analogy(find_best_analogy(x,d1,d2,filter_list,
                                      cluster_mode=cluster_mode,
                                      rmax=rmax,vmax=vmax,knn_filter=knn_filter))
    print("time: %.7f"%(time.time() - start))

start = time.time()
d1 = DomainLoader("analogy/data files/music_10k",cachefile="cache/music10k_cache.pkl").domain
print("d1 load time: %.7f"%(time.time() - start))
start = time.time()
d2 = DomainLoader("analogy/data files/programming_30k",cachefile="cache/p30k_cache.pkl").domain
print("d2 load time: %.7f"%(time.time() - start))

fba("C (programming language)",d2,d2,4,knn_filter=100)


def make_dendrograms(domain):
    import numpy as np
    from scipy.spatial import cKDTree

    lookup = {}
    data = []

    # ========================== graph the features ============


    featurepool1 = [x for x in domain.nodes.values() if x.knowledge_level > 10]

    random.shuffle(featurepool1)

    i = 0
    for f in sorted(featurepool1, key=lambda x:x.knowledge_level, reverse=True):
        lookup[i] = f.name
        i+=1
        data.append(domain.node_vectors[f.name])

    from scipy.cluster.hierarchy import fcluster, dendrogram, linkage, to_tree

    Z = linkage(data, 'complete')

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    from matplotlib import pyplot as plt

    for method in ['ward','complete','single','average','weighted','centroid','median']:
        Z = linkage(data, method)

        w = len(data)/10
        w = 25 if w < 25 else w

        h = len(data)/100
        h = 10 if h < 10 else h

        plt.figure(figsize=(w,h))
        plt.title('Hierarchical Clustering Dendrogram (method = %s)'%method)
        plt.xlabel('Concept')
        plt.ylabel('Distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            labels=[lookup[x] for x in range(len(data))],
        )

        plt.savefig('figures/dendrogram_%s.png'%method)
        plt.gcf().clear()


#make_dendrograms(d1)

print("Done")