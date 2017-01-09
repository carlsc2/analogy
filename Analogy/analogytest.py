from analogy import AIMind
from pprint import pprint
import random

import time

from cProfile import run

from itertools import combinations, product


#a1 = AIMind("data files/big_unrestricted_techdata.xml")
#a2 = AIMind("data files/big_unrestricted_music.xml")


#a2 = AIMind("data files/olympics_parsed.xml")
#a1 = AIMind("data files/plang_small.xml")
#a2 = AIMind("data files/music_small.xml")
#a1 = AIMind("data files/techdata_verbose.xml")
#a1 = AIMind("data files/plang_small_new_test.xml")


a1 = AIMind("data files/roman_empire_1000.xml")
a2 = AIMind("data files/ww2.xml")

#a1 = AIMind("data files/connected_plang_extended.xml")
#a2 = AIMind("data files/big_unrestricted_music_extended.xml")



def main():
    start = time.time()

    def fba(x,a1,a2=None,filter_list=None):
        a2 = a1 if not a2 else a2
        tmp = a1.find_best_analogy(x,a2,filter_list=filter_list,rmax=1,vmax=1)
        pprint(tmp)
        ex = a1.explain_analogy(tmp)
        print(ex)
        pprint(ex)


    #t = a1.get_analogy("Byzantine Empire", "Nazi Germany", a2)
    #pprint(t)
    #pprint(a1.explain_analogy(t))

    #fba("Adolf Hitler",a2,a2)
    #fba("Winston Churchill",a2,a1)

    #fba("Trajan",a1,a1)


    #fba("Byzantine Empire",a1,a2)
    #fba("Nazi Germany",a2,a1)

    fba("Adolf Hitler",a2,a1)

    #fba("Augustus",a1,a2)

    #fba("Rock music",a2,a1)

    #fba("Python (programming language)",a1,a1)
    #fba("Python (programming language)",a1,a2)

    #for f in nodes:
    #    fba(f,a1,a1,nodes)

    #fba("Rock music",a2,a1)
   

    print("time: %.7f"%(time.time() - start))


#pprint(a1.find_best_analogy_chain("Python (programming language)",a2,2))

#print("Done")

#a1 = AIMind("data files/plang_small.xml")

#a1 = AIMind("data files/techdata.xml")



#a2 = AIMind("data files/music.xml")

#pprint(a1.find_best_analogy("Python (programming language)",a2))

#pprint(a1.get_analogy("C (programming language)","Rock music",a2))

#pprint(a1.get_analogy("C (programming language)","Blues",a2))

#pprint(a1.get_analogy("Python (programming language)", "Electric blues", a2))
#pprint(a1.get_analogy("Python (programming language)", "Psychedelic rock", a2))


#a1 = AIMind("data files/roman_empire_1000.xml")
#with open("outdata_sorted.txt","w+",encoding="utf8") as f:
#    li = []

#    flist1 = [f.name for f in a1.features.values() if f.knowledge_level > 10]
#    flist2 = [f.name for f in a2.features.values() if f.knowledge_level > 10]

#    for f1 in flist1:
#        for f2 in flist2:
#            if f1 != f2:
#                x = a1.get_analogy(f1,f2,a2)
#                if x != None:
#                    li.append(x)
#    li = sorted(li, key=lambda x:x[0])
#    for x in li:
#        pprint(x,f)
#    print("Done")



#a2 = AIMind("data files/big_music.xml")


#a1 = AIMind("data files/music.xml")

#a2 = AIMind("data files/music_small.xml")

#a1 = AIMind("data files/roman_empire_500.xml")

#pprint(a1.find_best_analogy("C (programming language)",a2))
#pprint(a1.find_best_analogy("C++",a2))
#pprint(a1.find_best_analogy("C (programming language)",a1))

#pprint(a2.find_best_analogy("Rock music",a1))


#tmp = [a1.find_best_analogy(f,a2) for f in a1.features]
#pprint(sorted(tmp,key=lambda x:x[0]))


#a1.find_optimal_matchups(a2)

#pprint(sorted(a1.features.values(),key=lambda x:x.value))



#tmp = [a1.get_analogy(f1,f2,a1) for f1 in a1.features for f2 in a1.features if f1 != f2]
#tmp = [x for x in tmp if x != None]
#pprint(sorted(tmp,key=lambda x:x[0]))

#import pickle
#with open("data files/testdump.txt","rb") as f:
#    tmp = pickle.load(f)
#tmp2 = [x for x in tmp if len(x[4]) >= 3]
#pprint(sorted(tmp2,key=lambda x:len(x[4])))

#filterset = ['Commander-in-Chief, The Nore',
#             'Royal Navy',
#             'England',
#             'Environmental science',
#             'Midnight Sun (graphic novel)',
#             'Lewis Beaumont',
#             'Hugo Pearson',
#             'United States',
#             'Albert Hastings Markham',
#             'United Kingdom',
#             'London',
#             'Arctic exploration',
#             'HMS St Vincent (1815)']

#a1 = AIMind("data files/arctic_exploration_500.xml")

#tmp = [a1.find_best_analogy(f,a1) for f in filterset]
#pprint(sorted(tmp,key=lambda x:x[0]))

def make_dendrograms(graph):
    import numpy as np
    from scipy.spatial import cKDTree

    lookup = {}
    data = []

    # ========================== graph the features ============


    featurepool1 = list(graph.features.values())
    #featurepool2 = list(a2.features.values())
    random.shuffle(featurepool1)
    #random.shuffle(featurepool2)


    i = 0
    for f in sorted(featurepool1,key=lambda x:x.knowledge_level,reverse=True):
        if f.knowledge_level > 10 and i < 100:
            lookup[i] = f.name
            i+=1
            data.append(f.get_vector())



    # ========================== graph the rtypes ============

    #datapool = list(a1.rtype_index.items())
    #random.shuffle(datapool)

    #i = 0
    #for a,b in datapool:
    #    if i < 2500:
    #        lookup[i] = a
    #        i+=1
    #        data.append(b)


    #=========================================================

    #for f in featurepool2:
    #    #if f.knowledge_level > 15 and i < 200:
    #    if f.knowledge_level > 1:
    #        lookup[i] = f.name
    #        i+=1
    #        data.append(f.get_vector())




    #for j, f in enumerate(a2.features.values()):
    #    lookup[i+j+1] = f.name
    #    data.append(f.get_vector())

    #kdtree = cKDTree(data)

    #def get_similar_features(feature, number=10):
    #    distances, ndx = kdtree.query(a1.features[feature].get_vector(), number)
    #    for i,ix in enumerate(ndx):
    #        print("\t%r"%distances[i],lookup[ix])


    #def get_similar_feature_pairs(dist):
    #    for i1, i2 in kdtree.query_pairs(dist):
    #        print(lookup[i1], "::", lookup[i2])


    #get_similar_features("Google",50)

    #from numpy import array
    #from scipy.cluster.vq import vq, kmeans, whiten

    #whitened = whiten(data)

    #code_book = kmeans(whitened,15)[0]

    #decoded = vq(whitened,code_book)[0]

    #clusters = {}

    #for i,ix in enumerate(decoded):
    #    clusters.setdefault(ix,[]).append(lookup[i])

    #pprint(clusters)




    #================= hierarchy code =====================




    from scipy.cluster.hierarchy import fcluster, dendrogram, linkage, to_tree



    #Z = linkage(data, 'ward')
    #Z = linkage(data, 'complete')
    #Z = linkage(data, 'single')
    #Z = linkage(data, 'average')
    #Z = linkage(data, 'weighted')
    #Z = linkage(data, 'centroid')
    #Z = linkage(data, 'median')


    ##tree = to_tree(Z)

    ##def get_all_children(node,clist):
    ##    if node is None:
    ##        return
    ##    if node.is_leaf():
    ##        clist.append(lookup[node.id])
    ##    else:
    ##        get_all_children(node.left,clist)
    ##        get_all_children(node.right,clist)

    ##def build_hierarchy(node,tmp,depth,maxdepth):
    ##    if node is None:
    ##        return

    ##    if node.is_leaf():
    ##        tmp.append(lookup[node.id])
    ##    else:
    ##        lli = []
    ##        rli = []
    ##        tmp.append([node.dist,lli,rli])
    ##        if depth < maxdepth: #traverse hierarchy
    ##            build_hierarchy(node.left, lli, depth + 1, maxdepth)
    ##            build_hierarchy(node.right, rli, depth + 1, maxdepth)
    ##        else: #aggregate children
    ##            get_all_children(node.left,lli)
    ##            get_all_children(node.right,rli)



    ##out = []
    ##build_hierarchy(tree,out,0,5)


    ##import json

    ##with open("output.txt","w+") as f:
    ##    f.write(json.dumps(out, indent=4))

    Z = linkage(data, 'complete')

    #out = {}
    #for i,c in enumerate(fcluster(Z,1,depth=2)):
    #    out.setdefault(c,[]).append(lookup[i])

    #pprint(out)


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
        plt.xlabel('Feature')
        plt.ylabel('Distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            labels=[lookup[x] for x in range(len(data))],
        )

        plt.savefig('figures/dendrogram_%s.png'%method)
        plt.gcf().clear()





    

if __name__ == "__main__":
    #run('main()',sort=1)
    main()
    #make_dendrograms(a1)

    print("Done")