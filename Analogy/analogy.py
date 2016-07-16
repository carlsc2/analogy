import xml.etree.ElementTree as ET
from pprint import pprint

from itertools import product

from collections import Counter
from itertools import combinations, permutations, zip_longest

from math import sqrt

from urllib.request import urlopen
from urllib.parse import quote
import json
import pickle

import re


import numpy as np
from scipy import spatial
from numpy.linalg import norm

pattern = re.compile('\[/r/(.+)/,/c/en/([^/]*).*/c/en/([^/]*)')

JACCARD_DIMENSIONS = 28
#JACCARD_DIMENSIONS = 6



ENABLE_CONCEPTNET = False

if ENABLE_CONCEPTNET:
    from conceptnet5 import nodes
    from conceptnet5 import query
    from conceptnet5.uri import split_uri
    from conceptnet5.readers import dbpedia
    from conceptnet5.language.token_utils import un_camel_case
    from conceptnet5.nodes import standardized_concept_uri



#NULL_VEC = (0,)*(JACCARD_DIMENSIONS)
NULL_VEC = np.zeros(JACCARD_DIMENSIONS)
NULL_VEC2 = np.zeros(JACCARD_DIMENSIONS*2)


def jaccard_index(a,b):
    if len(a) == len(b) == 0:
        return 1
    return len(a&b) / len(a|b)

#def dice_coefficient(a,b):
#    total = (len(a) + len(b))
#    if total == 0:
#        return 1
#    overlap = len(a & b)
#    return overlap * 2.0/total

similarity_cache = {}

def euclidean_distance(v1,v2,_f=np.sum):  
    return sqrt(_f((v1-v2)**2))

def cosine_similarity(v1,v2):
    key = (v1.data.tobytes(),v2.data.tobytes())
    if key in similarity_cache:
        return similarity_cache[key]
    else:
        nu = sqrt(v1.dot(v1))
        nv = sqrt(v2.dot(v2))
        if nu == 0 or nv == 0:
            value = 0
        else:
            similarity = 2.0 - v1.dot(v2) / (nu * nv) #-1 to 1
            value = (similarity + 1) / 2 #0 to 1
        similarity_cache[key] = value
        return value

class Feature:
    def __init__(self,name,domain):
        self.name = name
        self.domain = domain
        self.outgoing_relations = set() #set of relations to other features
        self.incoming_relations = set() #set of relations from other features

        self.predecessors = set() #set of incoming features
        self.connections = set() #set of outgoing features

        self.rtypes = set() #set of outgoing relation types

        self.knowledge_level = len(self.outgoing_relations) + len(self.incoming_relations)

        self.value = 0

        self._vector = None
        self._vector2 = None

        self.text = ""

    def add_predecessor(self, rtype, pred):
        self.incoming_relations.add((rtype,pred))
        self.predecessors.add(pred)
        self.knowledge_level = len(self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None
        self._vector2 = None

    def add_relation(self, rtype, dest):
        self.connections.add(dest)
        self.outgoing_relations.add((rtype,dest))
        self.rtypes.add(rtype)
        self.knowledge_level = len(self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None
        self._vector2 = None

    def get_vector(self):
        ''' construct vector from centroid of rtypes '''
        if self._vector is None:#cache optimization
            tmp1 = [self.domain.rtype_index[rtype] for rtype,dest in self.outgoing_relations] or [NULL_VEC]
            tmp2 = [self.domain.rtype_index[rtype] for rtype,prev in self.incoming_relations] or [NULL_VEC]
            self._vector = np.concatenate((np.asarray(tmp1).mean(axis=0),
                                           np.asarray(tmp2).mean(axis=0))) #should be a unit vector
        return self._vector

    def get_vector2(self):
        ''' construct vector from neighbor base vectors '''
        if self._vector2 is None:#cache optimization
            tmp1 = [self.domain.features[dest].get_vector() for rtype,dest in self.outgoing_relations] or [NULL_VEC2]
            tmp2 = [self.domain.features[prev].get_vector() for rtype,prev in self.incoming_relations] or [NULL_VEC2]
            self._vector2 = np.concatenate((np.asarray(tmp1).mean(axis=0),
                                            np.asarray(tmp2).mean(axis=0))) #should be a unit vector
        return self._vector2


    def __repr__(self):
        return "<%s>(%d,%.d)"%(self.name,self.knowledge_level,self.value)

class AIMind:
    def __init__(self,filename=None,rawdata=None):
        self.features = {}
        self.usage_map = {}

        if filename:
            tree = ET.parse(filename)
        elif rawdata:
            tree = ET.ElementTree(ET.fromstring(rawdata))
        else:
            raise Exception("No data given")
        root = tree.getroot()
        features = root.find("Features")

        self.feature_id_table = {}

        #map all feature ids to name
        for feature in features.iter('Feature'):
            self.feature_id_table[feature.attrib["id"]] = feature.attrib["data"]

        #build relation structure
        for feature in features.iter('Feature'):
            fobj = Feature(feature.attrib["data"],self)
            speak = feature.find('speak')
            fobj.text = speak.text
            neighbors = feature.find('neighbors')
            for neighbor in neighbors.iter('neighbor'):
                fobj.add_relation(neighbor.attrib['relationship'],
                                  self.feature_id_table[neighbor.attrib['dest']])
            self.features[fobj.name] = (fobj)

        #map feature name to id
        self.r_feature_id_table = {b:a for a,b in self.feature_id_table.items()}

        for feature in self.features.values():
            for rtype, dest in feature.outgoing_relations:
                self.usage_map.setdefault(rtype,set()).add((feature.name,dest))
                self.features[dest].add_predecessor(rtype, 
                                                    feature.name)

        def ival(v,visited):#incoming values
            visited.add(v)
            return 1 + sum([ival(w,visited) for w in self.features[v].predecessors if not w in visited])

        def oval(v,visited):#outgoing values
            visited.add(v)
            return 1 + sum([oval(w,visited) for w in self.features[v].connections if not w in visited])

        #calculate values
        for feature in self.features:
            iv = ival(feature,set())
            ov = oval(feature,set())
            self.features[feature].value = iv * ov

        
        if ENABLE_CONCEPTNET:
            #augment knowledge with conceptnet
            self._augment_knowledge()
            

        #calculate rtype jaccard index
        self.rtype_index = self.index_rtypes()

    def get_id(self,feature):
        return self.r_feature_id_table[feature]

    def get_feature(self,fid):
        return self.feature_id_table[fid]

    def _augment_knowledge(self):
        try:
            with open("conceptnetquerycache.pkl","rb") as f:
                cache = pickle.load(f)
        except:
            cache = {}

        #use conceptnet to augment knowledge
        def get_results(feature):
            feature = feature.lower()

            if feature in cache:
                ret = cache[feature]
            else:
                with urlopen('http://conceptnet5.media.mit.edu/data/5.4%s?limit=1000'%quote('/c/en/'+feature)) as response:
                    html = response.read().decode('utf8')
                    result = json.loads(html)
                    ret = []
                    for x in result['edges']:
                        r = pattern.match(x['uri'][3:])
                        if r:
                            ret.append(r.groups())
                cache[feature] = ret
            return ret

        current_features = list(self.features)

        for feature in current_features:
            #convert dbpedia entry to conceptnet uri
            pieces = dbpedia.parse_topic_name(feature)
            pieces[0] = un_camel_case(pieces[0])
            cneturi = standardized_concept_uri('en', *pieces)

            ret = get_results(cneturi)
            for (rtype, src, dest) in ret:
                if src not in self.features:
                    self.features[src] = Feature(src,self)
                if dest not in self.features:
                    self.features[dest] = Feature(dest,self)

                self.usage_map.setdefault(rtype,set()).add((src,dest))
                self.features[src].add_relation(rtype, dest)
                self.features[dest].add_predecessor(rtype, src)

        with open("conceptnetquerycache.pkl","wb") as f:
            pickle.dump(cache,f)

    def explain_analogy(self, analogy, verbose=False):
        #only explain main relation
        if not analogy:
            return

        nrating, rating, total_rating, (src,trg), rassert, mapping = analogy

        narrative = ""
        narrative += "\t%s is like %s. "%(src,trg)

        narrative += "This is because"
        nchunks = []

        mentioned = set()

        for (a,b),(c,d,e,f) in mapping.items():
            if not verbose and a in mentioned:
                continue
            nchunks.append((src,a,b,trg,c,d))
            mentioned.add(a)
        for i,nc in enumerate(nchunks):
            a,b,c,d,e,f = nc
            if i == len(nchunks)-1:
                narrative += " and %s <%s> %s in the same way that %s <%s> %s.\n"%(a,b,c,d,e,f)
            else:
                narrative += " %s <%s> %s in the same way that %s <%s> %s,"%(a,b,c,d,e,f)
        return narrative

    def index_rtypes(self):
        hm = {} #aggregate rtypes across all usages
        for fnode in self.features.values():
            for (rtype,dest) in fnode.outgoing_relations:
                loses = fnode.rtypes - self.features[dest].rtypes
                gains = self.features[dest].rtypes - fnode.rtypes
                same = self.features[dest].rtypes & fnode.rtypes
                diff = self.features[dest].rtypes ^ fnode.rtypes
                lco,gco,smo,dfo,lci,gci,smi,dfi = hm.setdefault(rtype,(Counter(),Counter(),Counter(),Counter(),Counter(),Counter(),Counter(),Counter()))
                for r in loses:
                    lco[r] += 1
                for r in gains:
                    gco[r] += 1
                for r in same:
                    smo[r] += 1
                for r in diff:
                    dfo[r] += 1

            for (rtype,src) in fnode.incoming_relations:
                loses = fnode.rtypes - self.features[src].rtypes
                gains = self.features[src].rtypes - fnode.rtypes
                same = self.features[src].rtypes & fnode.rtypes
                diff = self.features[src].rtypes ^ fnode.rtypes
                lco,gco,smo,dfo,lci,gci,smi,dfi = hm.setdefault(rtype,(Counter(),Counter(),Counter(),Counter(),Counter(),Counter(),Counter(),Counter()))
                for r in loses:
                    lci[r] += 1
                for r in gains:
                    gci[r] += 1
                for r in same:
                    smi[r] += 1
                for r in diff:
                    dfi[r] += 1

        out = {} #compute metrics from rtypes
        for rtype, (lco, gco, smo, dfo, lci, gci, smi, dfi) in hm.items():
            x1 = set(lco)
            y1 = set(gco)
            z1 = set(smo)
            w1 = set(dfo)

            x2 = set(lci)
            y2 = set(gci)
            z2 = set(smi)
            w2 = set(dfi)

            score = (jaccard_index(x1,y1),
                     jaccard_index(x1,z1),
                     jaccard_index(x1,w1),
                     jaccard_index(x1,x2),
                     jaccard_index(x1,y2),
                     jaccard_index(x1,z2),
                     jaccard_index(x1,w2),
                     jaccard_index(y1,z1),
                     jaccard_index(y1,w1),
                     jaccard_index(y1,x2),
                     jaccard_index(y1,y2),
                     jaccard_index(y1,z2),
                     jaccard_index(y1,w2),
                     jaccard_index(z1,w1),
                     jaccard_index(z1,x2),
                     jaccard_index(z1,y2),
                     jaccard_index(z1,z2),
                     jaccard_index(z1,w2),
                     jaccard_index(w1,x2),
                     jaccard_index(w1,y2),
                     jaccard_index(w1,z2),
                     jaccard_index(w1,w2),
                     jaccard_index(x2,y2),
                     jaccard_index(x2,z2),
                     jaccard_index(x2,w2),
                     jaccard_index(y2,z2),
                     jaccard_index(y2,w2),
                     jaccard_index(z2,w2))

            out[rtype] = np.asarray(score, dtype=np.float)
        return out

    def get_analogy(self, src_feature, target_feature, target_domain):
        """Get the best analogy between two arbitrary features"""

        #ensure features exist
        if (not src_feature in self.features):
            print("Feature %s not in source domain"%src_feature)
            return None
        if (not target_feature in target_domain.features):
            print("Feature %s not in target domain"%target_feature)
            return None

        src_node = self.features[src_feature]
        svec = src_node.get_vector()

        c_node = target_domain.features[target_feature]
        cvec = c_node.get_vector()

        hypotheses = set()

        for r2,d2 in c_node.outgoing_relations: #for each pair in candidate
            for r1,d1 in src_node.outgoing_relations:#find best rtype to compare with

                #==== using cosine similarity ====
                rdiff = cosine_similarity(self.rtype_index[r1], target_domain.rtype_index[r2])
                d1vec = self.features[d1].get_vector()
                d2vec = target_domain.features[d2].get_vector()
                diff1 = svec-d1vec
                diff2 = cvec-d2vec
                vdiff = cosine_similarity(diff1, diff2)
                actual_score = (rdiff + vdiff)
                tscore = 2

                '''#==== using euclidean distance ====
                rdiff = 2-euclidean_distance(self.rtype_index[r1], target_domain.rtype_index[r2])
                d1vec = self.features[d1].get_vector()
                d2vec = target_domain.features[d2].get_vector()
                diff1 = svec-d1vec
                diff2 = cvec-d2vec
                vdiff = 4-euclidean_distance(diff1, diff2)
                actual_score = (rdiff + vdiff)
                tscore = 6'''

                hypotheses.add((actual_score / tscore, r1, d1, r2, d2, tscore, True))

        for r2,d2 in c_node.incoming_relations: #for each pair in candidate
            for r1,d1 in src_node.incoming_relations:#find best rtype to compare with

                #==== using cosine similarity ====
                rdiff = cosine_similarity(self.rtype_index[r1], target_domain.rtype_index[r2])
                d1vec = self.features[d1].get_vector()
                d2vec = target_domain.features[d2].get_vector()
                diff1 = svec-d1vec
                diff2 = cvec-d2vec
                vdiff = cosine_similarity(diff1, diff2)
                actual_score = (rdiff + vdiff)
                tscore = 2

                '''#==== using euclidean distance ====
                rdiff = 2-euclidean_distance(self.rtype_index[r1], target_domain.rtype_index[r2])
                d1vec = self.features[d1].get_vector()
                d2vec = target_domain.features[d2].get_vector()
                diff1 = svec-d1vec
                diff2 = cvec-d2vec
                vdiff = 4-euclidean_distance(diff1, diff2)
                actual_score = (rdiff + vdiff)
                tscore = 6'''

                hypotheses.add((actual_score / tscore, r1, d1, r2, d2, tscore, False))

        rassert = {} 
        hmap = {}
        best = {}
        rating = 0
        total_rating = 0

        #for each mh, pick the best then pick the next best non-conflicting
        for score,r1,src,r2,target,tscore,outgoing in sorted(hypotheses,reverse=True):
            score = score * tscore
            if (hmap.get(src) == target) or (src not in hmap.keys() and target not in hmap.values()):
                if r1 != r2 and r1 not in rassert.keys() and r2 not in rassert.values():
                    if r1 not in c_node.rtypes and\
                       r2 not in src_node.rtypes: #prevent crossmatching
                        rassert[r1] = r2
                if src not in hmap.keys() and target not in hmap.values():
                    hmap[src] = target
                    total_rating += tscore
                if r1 == r2 or rassert.get(r1) == r2:
                    best[(outgoing, r1, src)] = (r2, target, score, score/tscore)
                    rating += score
                else: #penalize inconsistent rtype matchup
                    total_rating += tscore


        #penalize score for non-matches
        for destobj in src_node.connections:
            if destobj not in hmap.keys():
                total_rating += 2#self.features[destobj].value

        for destobj in c_node.connections:
            if destobj not in hmap.values():
                total_rating += 2#target_domain.features[destobj].value

        #max1 = jaccard_index(src_node.connections, set(hmap.keys()))
        #max2 = jaccard_index(c_node.connections, set(hmap.values()))

        #rating = rating * (max1 + max2) / 2

        if total_rating == 0: #prevent divide by zero error
            return None

        normalized_rating = rating/total_rating

        return (normalized_rating,rating,total_rating,(src_feature,target_feature),rassert,best)
        #return (rating, normalized_rating, total_rating, (src_feature,target_feature), rassert, best)





    def find_best_analogy(self, src_feature, target_domain, filter_list=None):
        """
        Finds the best analogy between a specific feature in the source domain and any feature in the target domain.

        If filter_list is specified, only the features in that list will be selected from to make analogies.
        
        """

        candidate_pool = filter_list if filter_list != None else target_domain.features
        candidate_results = []

        for c_feature in candidate_pool:
            if target_domain == self and c_feature == src_feature:#find novel within same domain
                continue
            result = self.get_analogy(src_feature, c_feature, target_domain)
            if result:
                candidate_results.append(result)

        if not candidate_results:
            return None
        else:
            return sorted(candidate_results,key=lambda x:x[0])[-1]#best global analogy


    def find_best_analogy_chain(self, src_feature, target_domain, max_depth, depth=0, map_assert={}, results=[], visited = set()):
        """
        Finds the best analogy between a specific feature in the source domain and any feature in the target domain.
        Uses a chain of analogies to guarantee consistency.

        """

        if depth > max_depth:
            return None

        visited.add(src_feature)

        candidate_results = []

        for c_feature in target_domain.features:
            if target_domain == self and c_feature == src_feature:#find novel within same domain
                continue
            result = self.get_analogy(src_feature, c_feature, target_domain)
            if result:
                candidate_results.append(result)

        
        alright = False
        for candidate_result in sorted(candidate_results,key=lambda x:x[0],reverse=True):
            tmpr = candidate_result[5]
            good = True
            for (r1,d1), (r2,d2,s,t) in tmpr.items():
                if d1 in map_assert.keys() and map_assert[d1] != d2:
                    good = False
                    break

            if good:
                new_assert = map_assert.copy()
                new_results = []

                #new_assert.update(tmpr)
                new_results.append(candidate_result)

                tempr = []

                all_good = True
                for (r1,d1), (r2,d2,s,t) in tmpr.items():

                    new_assert[d1] = d2

                    if d1 in visited:
                        continue
                    tmpv = self.find_best_analogy_chain(d1, target_domain, max_depth, depth+1, new_assert, new_results, visited)
                    if tmpv == False:
                        return False
                    elif tmpv != None:
                        double_good,na,nr = tmpv
                    else:
                        continue                    
                    
                    if double_good:
                        tempr.append((na,nr))
                    else:
                        all_good = False
                        break

                if all_good:
                    for na,nr in tempr:
                        new_assert.update(na)
                        new_results.extend(nr)
                    return (True, new_assert, new_results)

        return False


        

    def get_all_analogies(self, src_feature, target_domain, filter_list=None):
        """
        Returns all analogies between a specific feature in the source domain and all features in the target domain.

        If filter_list is specified, only the features in that list will be selected from to make analogies.
        
        """

        candidate_pool = filter_list if filter_list != None else target_domain.features
        results = []

        for target_feature in candidate_pool:
            if target_domain == self and target_feature == src_feature:#find novel within same domain
                continue
            result = self.get_analogy(src_feature, target_feature, target_domain)
            if result:
                results.append(result)
        return results
