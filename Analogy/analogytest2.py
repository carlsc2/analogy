from analogy2 import make_analogy, find_best_analogy, explain_analogy
from utils.utils import Domain, AIMind, kulczynski_2
from pprint import pprint
import random

import time

from cProfile import run

import time

#from lxml import etree
#from SPARQLWrapper import SPARQLWrapper, JSON
#import re

#a1 = AIMind("data files/roman_empire_1000.xml")
#a1 = AIMind("data files/connected_plang.xml")

#a1.make_analogy()

#print(a1.get_closest_feature(a1.features['Diocletian'].get_vector(),5))




#a2 = AIMind("data files/ww2.xml")

##a1 = AIMind("data files/smallplangtest.xml")

##rx = [x for x in a1.features.values() if x.knowledge_level > 14 ]
##rxn = {x.name for x in rx}
###rx = sorted(rx,key=lambda x: x.knowledge_level)
###rx.reverse()

##idmap = {x.name:str(i+1) for i,x in enumerate(rx)}

##from lxml import etree

##root = etree.Element('AIMind')
##tmp = etree.Element('Root')
##tmp.attrib['id'] = '1'
##root.append(tmp)

##ftrs = etree.Element('Features')
##id = 1
##for f in rx:
##    ftr = etree.Element('Feature')
##    ftr.attrib['data'] = f.name
##    ftr.attrib['id'] = idmap[f.name]
##    neighbors = etree.Element('neighbors')
##    speak = etree.Element('speak')
##    speak.text = f.text
##    ftr.append(speak)
##    for rtype, dest in f.outgoing_relations:
##        if dest in rxn:
##            neighbor = etree.Element('neighbor')
##            neighbor.attrib['dest'] = idmap[dest]
##            neighbor.attrib['relationship'] = rtype
##            neighbors.append(neighbor)
##    ftr.append(neighbors)
##    ftrs.append(ftr)

##root.append(ftrs)



##with open("connected_plang.xml","wb+") as f:
##    f.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8'))



#def extend(datafile):

#    pattern = re.compile('(.*/)|(.*#)')
#    def clean(s):
#        return pattern.sub('',s).replace("_"," ")

#    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

#    graph = {}

#    def enhance(feature):
#        sparql.setQuery("""

#        CONSTRUCT WHERE {
#            <http://dbpedia.org/resource/%s> a ?c1 ; a ?c2 .
#            ?c1 rdfs:subClassOf ?c2 .
#            FILTER ( !strstarts(str(?c1), "http://dbpedia.org/class/yago/") )
#        }

#        """%feature.replace(" ","_"))

#        sparql.setReturnFormat(JSON)
#        results = sparql.query().convert()

#        for result in results["results"]["bindings"]:
#            graph.setdefault(clean(result['s']['value']),set()).add((clean(result['p']['value']),
#                                                           clean(result['o']['value'])))

#    a1 = AIMind("data files/%s"%datafile)

    
#    for feature in a1.features:
#        enhance(feature)

#    for k,v in graph.items():
#        try:
#            feature = a1.features[k]
#        except KeyError:
#            feature = Feature(k, a1)
#            a1.features[k] = feature
#        for r,d in v:
#            feature.add_relation(r,d)

#    rx = [x for x in a1.features.values()]
#    rxn = {x.name for x in a1.features.values()}

#    idmap = {x.name:str(i+1) for i,x in enumerate(rx)}

#    root = etree.Element('AIMind')
#    tmp = etree.Element('Root')
#    tmp.attrib['id'] = '1'
#    root.append(tmp)

#    ftrs = etree.Element('Features')
#    id = 1
#    for f in rx:
#        ftr = etree.Element('Feature')
#        ftr.attrib['data'] = f.name
#        ftr.attrib['id'] = idmap[f.name]
#        neighbors = etree.Element('neighbors')
#        speak = etree.Element('speak')
#        speak.text = f.text
#        ftr.append(speak)
#        for rtype, dest in f.outgoing_relations:
#            if dest in rxn:
#                neighbor = etree.Element('neighbor')
#                neighbor.attrib['dest'] = idmap[dest]
#                neighbor.attrib['relationship'] = rtype
#                neighbors.append(neighbor)
#        ftr.append(neighbors)
#        ftrs.append(ftr)

#    root.append(ftrs)



#    with open("data files/%s_extended.xml"%datafile,"wb+") as f:
#        f.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
#    print("DONE")


##extend("roman_empire_1000.xml")
#extend("big_unrestricted_music.xml")

a1 = AIMind("data files/roman_empire_1000.xml")
a2 = AIMind("data files/ww2.xml")


#a2 = AIMind("data files/testy.xml")

#a2 = AIMind("data files/ww2_1000_AIMind.xml")
#a1 = AIMind("data files/big_unrestricted_techdata.xml")
#a2 = AIMind("data files/connected_plang_extended.xml")

d1 = a1.as_domain()
d2 = a2.as_domain()

tmp = make_analogy("Augustus",d1,"Adolf Hitler",d2)
pprint(tmp)
#pprint(explain_analogy(tmp))

#pprint(make_analogy("Augustus",d1,"Augustus",d1))

def fba(x):
    start = time.time()
    tmp = find_best_analogy(x,d1,d2)
    pprint(tmp)
    print(explain_analogy(tmp))
    pprint(explain_analogy(tmp))
    print("time: %.7f"%(time.time() - start))


#fba("Augustus")

#for i in range(5):


opts = [k for k,v in d1.nodes.items() if v.knowledge_level > 5]
#while True:
#    x = random.choice(opts)
#    if d1.nodes[x].knowledge_level > 5:
#        fba(x)
#        break



t = []
for i in range(1):
    x = random.choice(opts)
    tmp = find_best_analogy(x,d1,d2)
    t.append(tmp)

best = max(t, key = lambda x: x["total_score"])
pprint(best)
print(explain_analogy(best))
pprint(explain_analogy(best))

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