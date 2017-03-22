from analogy2 import make_analogy, find_best_analogy, explain_analogy, get_all_analogies
from utils.utils import Domain, AIMind, DomainLoader
from pprint import pprint
import random

import time

from cProfile import run

import time

from utils import DBpediaCrawler

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

#with open("roman_10k_a3","w+") as f:
#    d1 = DBpediaCrawler.generate_graph(["Roman Empire"], 10000, None, None, None, None, True)
#    f.write(d1.serialize())


#start = time.time()
#d1 = DomainLoader("data files/songbird_10000",cachefile="cache/sb10k_cache.pkl").domain
#print("d1 load time: %.7f"%(time.time() - start))
'''
start = time.time()
d1 = DomainLoader("data files/ww2_10k_a3",cachefile="cache/ww210ka3_cache.pkl").domain
print("d2 load time: %.7f"%(time.time() - start))
start = time.time()
d2 = DomainLoader("data files/roman_10k_a3",cachefile="cache/r10ka3_cache.pkl").domain
print("d2 load time: %.7f"%(time.time() - start))
'''


start = time.time()
d1 = DomainLoader("data files/music_10k",cachefile="cache/music10k_cache.pkl").domain
print("d1 load time: %.7f"%(time.time() - start))
start = time.time()
d2 = DomainLoader("data files/programming_30k",cachefile="cache/p30k_cache.pkl").domain
print("d2 load time: %.7f"%(time.time() - start))

'''
start = time.time()
d1 = DomainLoader("data files/managed/dTQ4GEBJqT-jZYwc2r2cqQ==",
                  cachefile="cache/mg1.pkl").domain
print("d1 load time: %.7f"%(time.time() - start))
start = time.time()
d2 = DomainLoader("data files/managed/TTFDIqABJ6grPk7Uz2VhtQ==",
                  cachefile="cache/mg2.pkl").domain
print("d2 load time: %.7f"%(time.time() - start))

x1 = d1.get_closest_node(d2.node_vectors["Camping"],1)
x2 = d2.get_closest_node(d1.node_vectors["Hiking"],1)

pprint(x1)
pprint(x2)

process_analogy(make_analogy("Camping",d2,x1[0][1],d1))
process_analogy(make_analogy("Hiking",d1,x2[0][1],d2))
'''


m1 = np.mean(list(d1.node_vectors.values()),axis=0)
m2 = np.mean(list(d2.node_vectors.values()),axis=0)
offset = 0# m1 - m2


#x1 = d1.get_closest_node(d2.node_vectors["C (programming language)"] + offset,10)
#pprint(x1)
#European theatre of World War II

#x1 = d1.get_closest_node(d1.node_vectors["Mediterranean and Middle East theatre of World War II"] + offset, 30)
#pprint(x1)
#x2 = d2.get_closest_node(d1.node_vectors["Adolf Hitler"] + offset, 30)
#pprint(x2)
#x3 = d2.get_closest_node(d1.node_vectors["World War II"] + offset, 30)
#pprint(x3)

#pprint(d2.get_closest_relationship(d1.rtype_vectors["name"], 30))
#print(d1.rtype_vectors["name"])

#pprint(d1.get_closest_relationship(d1.rtype_vectors["name"], 30))


#fba("Mediterranean and Middle East theatre of World War II",d1,d1,0)


#fba("Adolf Hitler",d1,d1,0,knn_filter=100)
#pprint(make_analogy("Adolf Hitler",d1,"Adolf Hitler",d1))


#fba("Rock music",d1,d2,4,knn_filter=100)
fba("C (programming language)",d2,d2,4,knn_filter=100)

#opts = [k for k in d2.nodes if "programming" in k]

#results = []
#for x in opts:
#    tmp = make_analogy('Ginger (musician)',d1,x,d2)
#    results.append(tmp)

#results.sort(key = lambda x:x["total_score"],reverse=True)

#for x in results:
#    pprint(x)
#    print(explain_analogy(x))
#    pprint(explain_analogy(x))
#    print("\n")


#opts = [k for k,v in d1.nodes.items() if v.knowledge_level > 30]
#for i in range(20):
#    x = random.choice(opts)
#    if d1.nodes[x].knowledge_level > 300:
#        fba(x,d1,d2,1)
#    else:
#        fba(x,d1,d2,0)




#x = random.choice(opts)
#fba(x,d1,d2,3)


'''
import numpy as np


start = time.time()

cnode = d1.nodes["Rock music"]
m1 = np.mean(list(d1.node_vectors.values()),axis=0)
m2 = np.mean(list(d2.node_vectors.values()),axis=0)
offset = m1 - m2
clusters = []

for rtype in cnode.rtypes:
    cnds = [d1.node_vectors[d] for r,d
            in cnode.outgoing_relations if r == rtype]
    clusters.append(np.mean(cnds, axis=0))

for rtype in cnode.i_rtypes:
    cnds = [d1.node_vectors[d] for r,d
            in cnode.incoming_relations if r == rtype]
    clusters.append(np.mean(cnds, axis=0))

filter_list = []

for cluster in clusters:
    filter_list += [k for d,k in d2.get_closest_node(cluster-offset,10)]

#tmp = find_best_analogy("Rock music",d1,d2,filter_list=filter_list,cluster_mode=0)
#pprint(tmp)
#pprint(explain_analogy(tmp))
print("time: %.7f"%(time.time() - start))
'''


#fba("Yevgeny Shaposhnikov",d1,d2)
#fba("Adolf Hitler",d1,d2)
#fba("Joseph Stalin",d1,d2)
#fba("Ayn Rand",d1,d2)

#start = time.time()
##tmp = find_best_analogy("Adolf Hitler",d1,d2,restrict={"party"})
#tmp = find_best_analogy("Adolf Hitler",d1,d2)
#pprint(tmp)
#print(explain_analogy(tmp))
#pprint(explain_analogy(tmp))
#print("time: %.7f"%(time.time() - start))


#fba("Invasion of Poland",d1,d2,False)

'''
random.seed(234234234)
opts = [k for k,v in d2.nodes.items() if v.knowledge_level > 10]

start = time.time()
#filter_list = [x for x in d2.nodes if "battle" in x.lower()]
#filter_list = [x for d,x in d2.get_closest_node(d1.node_vectors["Invasion of Poland"],1000)]
filter_list = [random.choice(opts) for i in range(1000)]
#tmp = get_all_analogies("Invasion of Poland",d1,d2,filter_list=filter_list)
tmp = find_best_analogy("Invasion of Poland",d1,d2,filter_list=filter_list)
pprint(tmp)
pprint(explain_analogy(tmp))
print("time: %.7f"%(time.time() - start))
'''



#pprint(make_analogy("Invasion of Poland",d1,"Invasion of Poland",d1))

#for i in range(20):
#    
#    fba(random.choice(opts),d1,d2)

#run("fba(random.choice(opts),d1,d2)",sort=1)


#for i in range(30):
#    concept = random.choice(opts)
#    print(concept)
#    results = get_all_analogies(concept,d1,d2)

#    tmp1 = sorted(results, key=lambda x: x["total_score"], reverse=True)

#    ggmap = {x['target_concept']:i for i,x in enumerate(tmp1)}

#    for i in range(len(tmp1)):
#        ideals = d2.get_closest_node(d1.node_vectors["Yevgeny Shaposhnikov"],i+1)

#        #for i,(dist, name) in enumerate(ideals):
#            #print(i, ggmap[name], name)
#        done = False
#        for dist, name in ideals:
#            if ggmap[name] == 0:
#                done = True
#                break
#        if done:
#            print(i)
#            break




'''
#measure the percent of K nearest neighbors in the top K analogies


for i in range(1,len(tmp1)+1):
    ideals = d2.get_closest_node(d1.node_vectors["Adolf Hitler"],i)
    try:
        iset = {n for d,n in ideals}#the top i nearest neighbors
    except:
        iset = {ideals[1]}

    oset = {x['target_concept'] for x in tmp1[-i:]}

    print(i,len(iset&oset)/len(oset))
    '''

#dists = [(d2.node_vectors[x['target_concept']],x['total_score']) for x in testbed]





#tmp2 = sorted(candidate_results, key=lambda x: x["rating"])
#tmp3 = sorted(candidate_results, key=lambda x: x["confidence"])

#print("Best Rating")
#pprint(tmp2[-1])
#pprint(explain_analogy(tmp2[-1]))
#print("Best Confidence")
#pprint(tmp3[-1])
#pprint(explain_analogy(tmp3[-1]))





#from utils import permute_rtype_vector, cosine_similarity

##pprint(d2.get_closest_node(d2.node_vectors["Adolf Hitler"],5))
##pprint(d2.get_closest_node(d2.node_vectors["Joseph Stalin"],5))
#pprint(d2.get_closest_relationship(d2.rtype_vectors["predecessor"],15))
#pprint(d2.get_closest_relationship(permute_rtype_vector(d2.rtype_vectors["predecessor"]),15))


##print(cosine_similarity(d2.node_vectors["Adolf Hitler"],
##                        d2.node_vectors["Franklin D. Roosevelt"]))

#print(cosine_similarity(d2.rtype_vectors["predecessor"],
#                        d2.rtype_vectors["successor"]))

#print(cosine_similarity(d2.rtype_vectors["predecessor"],
#                        permute_rtype_vector(d2.rtype_vectors["successor"])))


##print(cosine_similarity(d2.rtype_vectors["before"],
##                        d2.rtype_vectors["after"]))

##print(cosine_similarity(d2.rtype_vectors["before"],
##                        permute_rtype_vector(d2.rtype_vectors["after"])))

'''
tmp = find_best_analogy("Pope Sixtus II",d1,d2)
pprint(tmp)
pprint(explain_analogy(tmp))
'''

'''
NOTE: <after> is same as <successor>

Pope Sixtus II is like Gabriel Auphan. This is because 'Pope Sixtus II'
<predecessor> 'Pope Stephen I' in the same way that 'François Darlan'
<after> 'Gabriel Auphan

--> implicitly compares 'Pope Stephen I' and 'François Darlan'

'Pope Sixtus II' <death place> 'Roman Empire' in the same way that 
'Gabriel Auphan' <country> 'Vichy France'

--> implicitly compares Roman Empire and Vichy France

'''

#Pope Sixtus II is like Gabriel Auphan
'''
tmp = make_analogy("Pope Sixtus II",d1,"Gabriel Auphan",d2)
pprint(tmp)
pprint(explain_analogy(tmp))
'''
'''
tmp = make_analogy("Roman Empire",d1,"Vichy France",d2)
pprint(tmp)
pprint(explain_analogy(tmp))
'''
'''
tmp = find_best_analogy("Vichy France",d2,d1)
pprint(tmp)
pprint(explain_analogy(tmp))
'''

#pprint(d2.get_closest_relationship(d2.rtype_vectors["after"],20))


'''
Good analogy between 'Yevgeny Shaposhnikov' and 'Titus'

... but not "best"

fix this
'''



def make_dendrograms(domain):
    import numpy as np
    from scipy.spatial import cKDTree

    lookup = {}
    data = []

    # ========================== graph the features ============


    featurepool1 = [x for x in domain.nodes.values() if x.knowledge_level > 10]

    #featurepool1 = [(x.knowledge_level,x,d1.node_vectors[x.name]) for x in d1.nodes.values() if x.knowledge_level > 10] + [(x.knowledge_level,x,d2.node_vectors[x.name]) for x in d2.nodes.values() if x.knowledge_level > 10]

    random.shuffle(featurepool1)

    i = 0
    #for l,f,v in sorted(featurepool1, key=lambda x:x[0], reverse=True):
    for f in sorted(featurepool1, key=lambda x:x.knowledge_level, reverse=True):
        lookup[i] = f.name
        i+=1
        data.append(domain.node_vectors[f.name])
        #data.append(v)

    from scipy.cluster.hierarchy import fcluster, dendrogram, linkage, to_tree

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