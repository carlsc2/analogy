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
        self.relations = set() #set of specific relations to other objects
        self.connections = set() #set of associated objects
        self.rtypes = set() #set of relation types between this and other objects
        self.knowledge_level = len(self.relations)

        self.predecessors = set() #inverse connections for fast lookup

        #for topo sort
        self.marked = False
        self.visited = False
        self.value = 0

        self._vector = None

    def add_predecessor(self,pred):
        self.predecessors.add(pred)
        self._vector = None

    def add_relation(self,rtype,dest):
        self.connections.add(dest)
        self.relations.add((rtype,dest))
        self.rtypes.add(rtype)
        self.knowledge_level = len(self.relations)
        self._vector = None

    def get_vector(self):
        if self._vector == None:#cache optimization
            #tmpl = [self.domain.rtype_index[rtype] for c1 in self.predecessors
            #                                  for c2 in self.domain.features[c1].connections
            #                                  for rtype in self.domain.features[c2].rtypes] or [self.domain.rtype_index[rtype] for rtype in self.rtypes]

            tmpl = [self.domain.rtype_index[rtype] for rtype in self.rtypes] or [self.domain.rtype_index[rtype] for c1 in self.predecessors
                                              for c2 in self.domain.features[c1].connections
                                              for rtype in self.domain.features[c2].rtypes]

            #tmpl = [self.domain.rtype_index[rtype] for rtype in self.rtypes]


            if len(tmpl):
                tmpv = tmpl[0]
                for rtype in tmpl[1:]:
                    tmpv = vadd(tmpv,rtype)
                self._vector = tmpv#normalize(tmpv)
            else:
                self._vector = (0,)*JACCARD_DIMENSIONS

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
            for rtype, dest in feature.relations:
                self.usage_map.setdefault(rtype,set()).add((feature.name,dest))
                self.features[dest].add_predecessor(feature.name)

        def val(v,visited):
            visited.add(v)
            return 1 + sum([val(w,visited) for w in self.features[v].predecessors if not w in visited])

        #calculate values
        for feature in self.features:
            self.features[feature].value = val(feature,set())

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
            for (rtype,dest) in fnode1.relations:
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

            out[rtype] = score#normalize(score)
        return out

    def get_analogy(self, src_feature, target_feature, target_domain):
        """Get the best analogy between two arbitrary features"""

        if not src_feature in self.features:
            return None

        ixmap = self.rtype_index.copy()
        #merge target domain and current
        #use highest score version for now
        #probably best to merge before jaccard
        for k,v in target_domain.rtype_index.items():
            if k in ixmap:
                ixmap[k] = max(ixmap[k],v,key=lambda x: sqmag(x))
            else:
                ixmap[k] = v

        src_node = self.features[src_feature]
        svec = src_node.get_vector()

        if target_feature == src_feature: #ignore analogies to self
            return None

        c_node = target_domain.features[target_feature]

        #keep track of the best result only
        bestrating = -10000
        bestresult = None
        hypotheses = set()

        cvec = c_node.get_vector()

        for r2,d2 in c_node.relations: #for each pair in candidate
            #scoreval = c_node.value + src_node.value#c_node.value/max2 + src_node.value/max1

            for r1,d1 in src_node.relations:#find best rtype to compare with

                #base score of matchup
                scoreval = self.features[d1].value + target_domain.features[d2].value

                #weight by strength of relation matchup
                rdiff = (JACCARD_DIMENSIONS-sq_edist(ixmap[r1],ixmap[r2]))/JACCARD_DIMENSIONS

                #weight by relative feature distance from parent
                d1vec = self.features[d1].get_vector()
                d2vec = target_domain.features[d2].get_vector()

                diff1 = vsub(svec,d1vec)
                diff2 = vsub(cvec,d2vec)
                vdiff = 1-sq_edist(diff1,diff2)

                actual_score = (scoreval * vdiff) + (scoreval * rdiff)

                hypotheses.add((actual_score / (scoreval*2), r1, d1, r2, d2, scoreval*2))

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
                    best[(r1,src)] = (r2,target)
                    rating += score
                else: #penalize inconsistent rtype matchup
                    total_rating += tscore


        #penalize score for non-matches
        for destobj in src_node.connections:
            if destobj not in hmap.keys():
                total_rating += self.features[destobj].value

        for destobj in c_node.connections:
            if destobj not in hmap.values():
                total_rating += target_domain.features[destobj].value

        if total_rating == 0: #prevent divide by zero error
            return None

        normalized_rating = rating/total_rating

        if normalized_rating > bestrating:
            return (normalized_rating,rating,total_rating,(src_feature,target_feature),rassert,best)
        else:
            return None



    def find_best_analogy(self, src_feature, target_domain, filter_list=None):

        if not src_feature in self.features:
            return None

        ixmap = self.rtype_index.copy()
        #merge target domain and current
        #use highest score version for now
        #probably best to merge before jaccard
        for k,v in target_domain.rtype_index.items():
            if k in ixmap:
                ixmap[k] = max(ixmap[k],v,key=lambda x: sqmag(x))
            else:
                ixmap[k] = v

        src_node = self.features[src_feature]
        svec = src_node.get_vector()

        candidate_pool = filter_list if filter_list != None else target_domain.features

        candidate_results = []

        for c_feature in candidate_pool:
            if c_feature == src_feature: #ignore analogies to self
                continue

            c_node = target_domain.features[c_feature]

            #keep track of the best result only
            bestrating = -10000
            bestresult = None
            hypotheses = set()

            cvec = c_node.get_vector()

            for r2,d2 in c_node.relations: #for each pair in candidate
                #scoreval = c_node.value + src_node.value#c_node.value/max2 + src_node.value/max1

                for r1,d1 in src_node.relations:#find best rtype to compare with

                    #base score of matchup
                    scoreval = self.features[d1].value + target_domain.features[d2].value

                    #weight by strength of relation matchup
                    rdiff = (JACCARD_DIMENSIONS-sq_edist(ixmap[r1],ixmap[r2]))/JACCARD_DIMENSIONS

                    #weight by relative feature distance from parent
                    d1vec = self.features[d1].get_vector()
                    d2vec = target_domain.features[d2].get_vector()

                    diff1 = vsub(svec,d1vec)
                    diff2 = vsub(cvec,d2vec)
                    vdiff = 1-sq_edist(diff1,diff2)

                    actual_score = (scoreval * vdiff) + (scoreval * rdiff)
                    tscore = scoreval*2

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
                        best[(r1,src)] = (r2,target)
                        rating += score
                    else: #penalize inconsistent rtype matchup
                        total_rating


            #penalize score for non-matches
            for destobj in src_node.connections:
                if destobj not in hmap.keys():
                    total_rating += self.features[destobj].value

            for destobj in c_node.connections:
                if destobj not in hmap.values():
                    total_rating += target_domain.features[destobj].value

            if total_rating == 0: #prevent divide by zero error
                continue

            normalized_rating = rating/total_rating

            if normalized_rating > bestrating:
                bestrating = normalized_rating
                bestresult = (normalized_rating,rating,total_rating,(src_feature,c_feature),rassert,best)

            if bestresult:
                candidate_results.append(bestresult)

        if not candidate_results:
            return None
        else:
            return sorted(candidate_results,key=lambda x:x[0])[-1]#best global analogy

        


    def find_optimal_matchups(self, target_domain):

        ixmap = self.rtype_index.copy()
        #merge target domain and current
        #use highest score version for now
        #probably best to merge before jaccard
        for k,v in target_domain.rtype_index.items():
            if k in ixmap:
                ixmap[k] = max(ixmap[k],v,key=lambda x: sqmag(x))
            else:
                ixmap[k] = v

        optimax = Counter()

        occurances = Counter()

        #srclen = len(src_node.relations)

        #max1 = max([f.value for f in self.features.values()])
        #max2 = max([f.value for f in target_domain.features.values()])

        for src_feature in self.features:
            src_node = self.features[src_feature]
            svec = src_node.get_vector()
            for c_feature in target_domain.features:

                c_node = target_domain.features[c_feature]

                cvec = c_node.get_vector()

                #clen = len(c_node.relations)



                for r2,d2 in c_node.relations: #for each pair in candidate
                    #scoreval = c_node.value + src_node.value#c_node.value/max2 + src_node.value/max1

                    #relative_score1 = target_domain.features[d2].value / max2


                    for r1,d1 in src_node.relations:#find best rtype to compare with

                        #relative_score2 = self.features[d1].value / max1

                        #base score of matchup
                        scoreval = self.features[d1].value + target_domain.features[d2].value

                        #weight by strength of relation matchup
                        #rdiff = (JACCARD_DIMENSIONS-sq_edist(ixmap[r1],ixmap[r2]))/JACCARD_DIMENSIONS
                        rdiff = sq_edist(ixmap[r1],ixmap[r2])

                        #weight by relative feature distance from parent
                        d1vec = self.features[d1].get_vector()
                        d2vec = target_domain.features[d2].get_vector()

                        diff1 = vsub(svec,d1vec)
                        diff2 = vsub(cvec,d2vec)
                        vdiff = sq_edist(diff1,diff2)

                        #diff1 = sq_edist(svec,d1vec)
                        #diff2 = sq_edist(cvec,d2vec)

                        #vdiff = abs(diff1-diff2)

                        #vdiff = abs(relative_score1-relative_score2) * rdiff
                       
                        actual_score = scoreval/(vdiff if vdiff != 0 else 1) + scoreval/(rdiff if rdiff != 0 else 1)#(scoreval * vdiff) + (scoreval * rdiff)

                        optimax[(r2,r1)] += actual_score
                        occurances[(r2,r1)] += 1

        tmpl = [(a,b) for a,b in optimax.most_common()]

        pprint(sorted(tmpl,key=lambda x:x[1]))

        #for (a,b),c in optimax.most_common():



