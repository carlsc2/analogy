from analogy2 import make_analogy, find_best_analogy
from utils import Domain, AIMind
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
#a1 = AIMind("data files/big_unrestricted_techdata.xml")
#a2 = AIMind("data files/connected_plang_extended.xml")

d1 = a1.as_domain()
d2 = a2.as_domain()

#pprint(make_analogy("Augustus",d1,"Adolf Hitler",d2))

start = time.time()
pprint(find_best_analogy("Augustus",d1,d2,rmax=1,vmax=1,threshold=0.75))
print("time: %.7f"%(time.time() - start))


#import numpy as np
#from utils import jaccard_index
#from math import sqrt

#def index_rtypes(nodes):
#    """Constructs vector representations for every type of relationship
#    in the domain.        
#    """
#    out = {}
#    for fnode in nodes.values():
#        for (rtype, dest) in fnode.outgoing_relations:
#            dnode = nodes[dest]
#            x1 = fnode.rtypes - dnode.rtypes
#            y1 = dnode.rtypes - fnode.rtypes
#            z1 = dnode.rtypes & fnode.rtypes
#            w1 = dnode.rtypes ^ fnode.rtypes

#            rval = out.setdefault(rtype,np.zeros(6))

#            score = np.array([jaccard_index(x1, y1),
#                              jaccard_index(x1, z1),
#                              jaccard_index(x1, w1),
#                              jaccard_index(y1, z1),
#                              jaccard_index(y1, w1),
#                              jaccard_index(z1, w1)], dtype=np.float)

#            # inverse rtype score is
#            # original: (0,1,2,3,4,5)
#            # inverse: (0,3,4,1,2,5)

#            out[rtype] = rval + score

#    #normalize everything

#    for r,v in out.items():
#        out[r] = v / sqrt(v.dot(v))

#    return out


#x = index_rtypes(d1.nodes)
#pprint(x)