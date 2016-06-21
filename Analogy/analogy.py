import xml.etree.ElementTree as ET
from pprint import pprint

from itertools import product

from collections import Counter
from itertools import combinations, permutations, zip_longest

from math import sqrt

JACCARD_DIMENSIONS = 6

def jaccard_index(a,b):
    if len(a) == len(b) == 0:
        return 1
    return len(a&b) / len(a|b)

def dice_coefficient(a,b):
    total = (len(a) + len(b))
    if total == 0:
        return 1
    overlap = len(a & b)
    return overlap * 2.0/total

def sqmag(v):
    return sum(x*x for x in v)

def mag(v):
    return sqrt(sum(x*x for x in v))

def sq_edist(v1,v2):
    return sum((x-y)*(x-y) for x,y in zip(v1,v2))

def vadd(v1,v2):
    return tuple(x+y for x,y in zip(v1,v2))

def vsub(v1,v2):
    return tuple(x-y for x,y in zip(v1,v2))

def normalize(v):
    magnitude = mag(v)
    if magnitude == 0:
        return v
    return tuple(x/magnitude for x in v)

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

        #for topo sort
        self.marked = False
        self.visited = False
        self.value = 0

        self._vector = None

    def add_predecessor(self, rtype, pred):
        self.incoming_relations.add((rtype,pred))
        self.predecessors.add(pred)
        self.knowledge_level = len(self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None

    def add_relation(self, rtype, dest):
        self.connections.add(dest)
        self.outgoing_relations.add((rtype,dest))
        self.rtypes.add(rtype)
        self.knowledge_level = len(self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None

    def get_vector(self):
        if self._vector == None:#cache optimization

            tmpl = [self.domain.rtype_index[rtype] for rtype in self.rtypes] + \
                   [self.domain.rtype_index[rtype] for rtype,c1 in self.incoming_relations]

            if len(tmpl):
                tmpv = tmpl[0]
                for rtype in tmpl[1:]:
                    tmpv = vadd(tmpv,rtype)
                self._vector = normalize(tmpv)
            else:
                self._vector = (0,)*(JACCARD_DIMENSIONS*2)

        return self._vector

    def __repr__(self):
        return "<%s>(%d,%.d)"%(self.name,self.knowledge_level,self.value)

class AIMind:
    def __init__(self,filename=None,rawdata=None):
        self.features = {}
        self.usage_map = {}


        self.topo_sorted_features = []

        if filename:
            tree = ET.parse(filename)
        elif rawdata:
            tree = ET.ElementTree(ET.fromstring(rawdata))
        else:
            raise Exception("No data given")
        root = tree.getroot()
        features = root.find("Features")
        relations = root.find("Relations")

        self.feature_id_table = {}

        #map all feature ids to name
        for feature in features.iter('Feature'):
            self.feature_id_table[feature.attrib["id"]] = feature.attrib["data"]

        #build relation structure
        for feature in features.iter('Feature'):
            fobj = Feature(feature.attrib["data"],self)
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

        #calculate rtype jaccard index
        self.rtype_index = self.index_rtypes()


    def get_id(self,feature):
        return self.r_feature_id_table[feature]

    def get_feature(self,fid):
        return self.feature_id_table[fid]

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

        for (a,b),(c,d) in mapping.items():
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
        for fnode1 in self.features.values():
            for (rtype,dest) in fnode1.outgoing_relations:
                loses = fnode1.rtypes - self.features[dest].rtypes
                gains = self.features[dest].rtypes - fnode1.rtypes
                same = self.features[dest].rtypes & fnode1.rtypes
                diff = self.features[dest].rtypes ^ fnode1.rtypes
                lc,gc,sm,df = hm.setdefault(rtype,(Counter(),Counter(),Counter(),Counter()))
                for r in loses:
                    lc[r] += 1
                for r in gains:
                    gc[r] += 1
                for r in same:
                    sm[r] += 1
                for r in diff:
                    df[r] += 1

        out = {} #compute metrics from rtypes
        for rtype, (lc, gc, sm, df) in hm.items():
            x = set(lc)
            y = set(gc)
            z = set(sm)
            w = set(df)

            score = (jaccard_index(x,y),
                     jaccard_index(x,z),
                     jaccard_index(x,w),
                     jaccard_index(y,z),
                     jaccard_index(y,w),
                     jaccard_index(z,w))
            
            #score = (dice_coefficient(x,y),
            #         dice_coefficient(x,z),
            #         dice_coefficient(x,w),
            #         dice_coefficient(y,z),
            #         dice_coefficient(y,w),
            #         dice_coefficient(z,w))

            out[rtype] = normalize(score)
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
                #base score of matchup
                ##iscore, oscore = vadd(self.features[d1].value, target_domain.features[d2].value)
                #scoreval = self.features[d1].value + target_domain.features[d2].value

                #weight by strength of matchup (if distance between them is 0, strength is 2)
                rdiff = 2-sq_edist(self.rtype_index[r1], target_domain.rtype_index[r2])

                #weight by relative feature distance from parent
                d1vec = self.features[d1].get_vector()
                d2vec = target_domain.features[d2].get_vector()

                diff1 = normalize(vsub(svec,d1vec))
                diff2 = normalize(vsub(cvec,d2vec))
                vdiff = 2-sq_edist(diff1,diff2)

                actual_score = (rdiff + vdiff) #*scoreval
                tscore = 4 #*scoreval

                hypotheses.add((actual_score / tscore, r1, d1, r2, d2, tscore))

        rassert = {} 
        hmap = {}
        best = {}
        rating = 0
        total_rating = 0

        #for each mh, pick the best then pick the next best non-conflicting
        for score,r1,src,r2,target,tscore in sorted(hypotheses,reverse=True):
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
                    best[(r1,src)] = (r2,target,score,score/tscore)
                    rating += score
                else: #penalize inconsistent rtype matchup
                    total_rating += tscore


        #penalize score for non-matches
        #for destobj in src_node.connections:
        #    if destobj not in hmap.keys():
        #        total_rating += 2#self.features[destobj].value

        #for destobj in c_node.connections:
        #    if destobj not in hmap.values():
        #        total_rating += 2#target_domain.features[destobj].value

        if total_rating == 0: #prevent divide by zero error
            return None

        normalized_rating = rating/total_rating

        #return (normalized_rating,rating,total_rating,(src_feature,target_feature),rassert,best)
        return (rating,normalized_rating,total_rating,(src_feature,target_feature),rassert,best)





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

        


    def find_optimal_matchups(self, target_domain):

        optimax = Counter()
        occurances = Counter()

        for src_feature in self.features:
            src_node = self.features[src_feature]
            svec = src_node.get_vector()

            #counter ratios
            srctmp = Counter()
            for rtype,dest in src_node.outgoing_relations:
                srctmp[rtype] += 1
            srctotal = sum(srctmp.values())
            src_ratios = {k:v/srctotal for k,v in srctmp.items()}

            for c_feature in target_domain.features:
                c_node = target_domain.features[c_feature]
                cvec = c_node.get_vector()

                #counter ratios
                ctmp = Counter()
                for rtype,dest in c_node.outgoing_relations:
                    ctmp[rtype] += 1
                ctotal = sum(ctmp.values())
                c_ratios = {k:v/ctotal for k,v in ctmp.items()}

                #ktotal = len(c_node.outgoing_relations) * len(src_node.outgoing_relations)

                for r2,d2 in c_node.outgoing_relations: #for each pair in candidate
                    for r1,d1 in src_node.outgoing_relations:#find best rtype to compare with
                        #base score of matchup
                        ##scoreval = self.features[d1].value + target_domain.features[d2].value
                        scoreval = self.features[d1].value * src_ratios[r1] + target_domain.features[d2].value * c_ratios[r2]

                        #weight by strength of relation matchup
                        #rdiff = (JACCARD_DIMENSIONS-sq_edist(self.rtype_index[r1],
                        #                                     target_domain.rtype_index[r2])) / JACCARD_DIMENSIONS
                        rdiff = sq_edist(self.rtype_index[r1],target_domain.rtype_index[r2])

                        #weight by relative feature distance from parent
                        d1vec = self.features[d1].get_vector()
                        d2vec = target_domain.features[d2].get_vector()

                        diff1 = normalize(vsub(svec,d1vec))
                        diff2 = normalize(vsub(cvec,d2vec))
                        #diff1 = vsub(svec,d1vec)
                        #diff2 = vsub(cvec,d2vec)

                        vdiff = sq_edist(diff1,diff2)

                        #actual_score = (scoreval * vdiff) + (scoreval * rdiff)
                        actual_score = vdiff + rdiff

                        optimax[(r2,r1)] += actual_score
                        occurances[(r2,r1)] += 1

        tmpl = [(a,b) for a,b in optimax.most_common()]

        pprint(sorted(tmpl,key=lambda x:x[1]))

        #for (a,b),c in optimax.most_common():
