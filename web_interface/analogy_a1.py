import xml.etree.ElementTree as ET
from collections import Counter
from math import sqrt, log
import numpy as np
from lru import LRU


JACCARD_DIMENSIONS = 6
NULL_VEC = np.zeros(JACCARD_DIMENSIONS)
NULL_VEC2 = np.zeros(JACCARD_DIMENSIONS * 2)

def kulczynski_2(a, b):
    '''Computes the Kulczynski-2 measure between two sets

    This is the arithmetic mean probability that if one object has an attribute,
    the other object has it too

    1 means completely similar, 0 means completely different.

    '''
    if len(a) == len(b) == 0:  # if both sets are empty, return 1
        return 1
    z = len(a & b)
    if z == 0:  # if the union is empty, the sets are disjoint, return 0
        return 0
    x = z / (len(a - b) + z)
    y = z / (len(b - a) + z)
    return (x + y) / 2


def jaccard_index(a, b):
    '''Computes the jaccard index between two sets. 
    
    1 means completely similar, 0 means completely different.'''
    if len(a) == len(b) == 0:  # if both sets are empty, return 1
        return 1
    n = len(a&b)
    return n / (len(a) + len(b) - n)


def dice_coefficient(a, b):
    '''Computes the dice coefficient between two sets

    1 means completely similar, 0 means completely different.'''
    total = (len(a) + len(b))
    if total == 0:
        return 1
    overlap = len(a & b)
    return overlap * 2.0 / total

similarity_cache = LRU(10000) #use LRU cache to limit size and avoid memory error


def euclidean_distance(v1, v2, _f=np.sum):
    return sqrt(_f((v1 - v2)**2))


def cosine_similarity(v1, v2):
    key = (v1.data.tobytes(), v2.data.tobytes())
    try:
        return similarity_cache[key]
    except KeyError:
        nu = v1.dot(v1)
        nv = v2.dot(v2)
        if nu == 0 or nv == 0:
            value = 0
        else:
            value = 0.5 * (v1.dot(v2) / sqrt(nu * nv) + 1)

        similarity_cache[key] = value
        return value


class Feature:

    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.outgoing_relations = set()  # set of relations to other features
        self.incoming_relations = set()  # set of relations from other features

        self.predecessors = set()  # set of incoming features
        self.connections = set()  # set of outgoing features

        self.rtypes = set()  # set of outgoing relation types

        self.rtype_count = Counter() #how many times each rtype is used

        self.knowledge_level = len(
            self.outgoing_relations) + len(self.incoming_relations)

        self._vector = None
        self._vector2 = None

        self.text = ""

    def get_rtype_ratios(self):
        total = sum(self.rtype_count.values())
        return {x:self.rtype_count[x]/total for x in self.rtype_count}


    def add_predecessor(self, rtype, pred):
        self.incoming_relations.add((rtype, pred))
        self.rtype_count[rtype] += 1
        self.predecessors.add(pred)
        self.knowledge_level = len(
            self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None
        self._vector2 = None

    def add_relation(self, rtype, dest):
        self.connections.add(dest)
        self.outgoing_relations.add((rtype, dest))
        self.rtypes.add(rtype)
        self.rtype_count[rtype] += 1
        self.knowledge_level = len(
            self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None
        self._vector2 = None

    def get_vector(self):
        ''' construct vector from centroid of rtypes '''
        if self._vector is None:  # cache optimization
            tmp1 = [self.domain.rtype_index[rtype]
                    for rtype, dest in self.outgoing_relations] or [NULL_VEC]
            tmp2 = [self.domain.rtype_index[rtype]
                    for rtype, prev in self.incoming_relations] or [NULL_VEC]
            self._vector = np.concatenate((np.asarray(tmp1).mean(axis=0),
                                           np.asarray(tmp2).mean(axis=0)))
        return self._vector

    def get_vector2(self):
        ''' construct vector from neighbor base vectors '''
        if self._vector2 is None:  # cache optimization
            tmp1 = [self.domain.features[dest].get_vector()
                    for rtype, dest in self.outgoing_relations] or [NULL_VEC2]
            tmp2 = [self.domain.features[prev].get_vector()
                    for rtype, prev in self.incoming_relations] or [NULL_VEC2]
            self._vector2 = np.concatenate((np.asarray(tmp1).mean(axis=0),
                                            np.asarray(tmp2).mean(axis=0)))
        return self._vector2

    def __repr__(self):
        return "<%s>(%d)" % (self.name, self.knowledge_level)


class AIMind:

    def __init__(self, filename=None, rawdata=None):
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

        # map all feature ids to name
        for feature in features.iter('Feature'):
            self.feature_id_table[feature.attrib["id"]] = feature.attrib["data"]

        # build relation structure
        for feature in features.iter('Feature'):
            fobj = Feature(feature.attrib["data"], self)
            tmp = feature.find('description')
            if tmp != None:
                fobj.text = tmp.text
            else:
                tmp = feature.find('speak')
                if tmp != None:
                    fobj.text = tmp.text
            neighbors = feature.find('neighbors')
            for neighbor in neighbors.iter('neighbor'):
                fobj.add_relation(
                    neighbor.attrib['relationship'],
                    self.feature_id_table[neighbor.attrib['dest']])
            self.features[fobj.name] = (fobj)

        # map feature name to id
        self.r_feature_id_table = {b: a for a,
                                   b in self.feature_id_table.items()}

        for feature in self.features.values():
            for rtype, dest in feature.outgoing_relations:
                self.usage_map.setdefault(rtype, set()).add((feature.name,
                                                             dest))
                self.features[dest].add_predecessor(rtype,
                                                    feature.name)
        # calculate rtype jaccard index
        self.rtype_index = self.index_rtypes()

    def get_id(self, feature):
        return self.r_feature_id_table[feature]

    def get_feature(self, fid):
        return self.feature_id_table[fid]

    def explain_analogy(self, src, trg, mapping, verbose=False):

        narrative = ""
        narrative += "\t%s is like %s. " % (src, trg)

        narrative += "This is because"
        nchunks = []

        mentioned = set()

        for (x, a, b), (c, d, e, f) in mapping.items():
            if not verbose and a in mentioned:
                continue
            if x == "INCOMING":
                nchunks.append((b, a, src, d, c, trg, f))
            if x == "OUTGOING":
                nchunks.append((src, a, b, trg, c, d, f))
            mentioned.add(a)
        for i, nc in enumerate(sorted(nchunks,key=lambda x:x[-1],reverse=True)):
            a, b, c, d, e, f, s = nc
            if i == len(nchunks) - 1:
                narrative += " and '%s' <%s> '%s' in the same way that '%s' <%s> '%s'.\n" % (
                    a, b, c, d, e, f)
            else:
                narrative += " '%s' <%s> '%s' in the same way that '%s' <%s> '%s'," % (
                    a, b, c, d, e, f)
        return narrative

    def index_rtypes(self):
        hm = {}  # aggregate rtypes across all usages
        for fnode in self.features.values():
            for (rtype, dest) in fnode.outgoing_relations:
                loses = fnode.rtypes - self.features[dest].rtypes
                gains = self.features[dest].rtypes - fnode.rtypes
                same = self.features[dest].rtypes & fnode.rtypes
                diff = self.features[dest].rtypes ^ fnode.rtypes

                if rtype not in hm:
                    hm[rtype] = (Counter(), Counter(), Counter(),
                                 Counter(), Counter())
                lco, gco, smo, dfo, gci = hm[rtype]

                for r in loses:
                    lco[r] += 1
                for r in gains:
                    gco[r] += 1
                for r in same:
                    smo[r] += 1
                for r in diff:
                    dfo[r] += 1

            #only incoming gains matters
            for (rtype, src) in fnode.incoming_relations:
                gains = self.features[src].rtypes - fnode.rtypes

                if rtype not in hm:
                    hm[rtype] = (Counter(), Counter(), Counter(),
                                 Counter(), Counter())
                lco, gco, smo, dfo, gci = hm[rtype]

                for r in gains:
                    gci[r] += 1


        out = {}  # compute metrics from rtypes
        for rtype, (lco, gco, smo, dfo, gci) in hm.items():

            #def adjust(x):  # eliminate outlier data for better results
            #    n = len(x)
            #    total_count = sum(x.values())
            #    return set(a for a, b in x.items() if b / total_count > 1/n)
            #x1 = adjust(lco)
            #y1 = adjust(gco)
            #z1 = adjust(smo)
            #w1 = adjust(dfo)
            #y2 = adjust(gci)

            x1 = set(lco)
            y1 = set(gco)
            z1 = set(smo)
            w1 = set(dfo)
            y2 = set(gci)

            # new ==> linearly independent columns only
            score = (jaccard_index(x1, y1),
                     jaccard_index(x1, z1),
                     jaccard_index(x1, w1),
                     jaccard_index(x1, y2),
                     jaccard_index(y1, z1),
                     jaccard_index(z1, w1))

            

            #score = (kulczynski_2(x1, y1),
            #        kulczynski_2(x1, z1),
            #        kulczynski_2(x1, w1),
            #        kulczynski_2(x1, y2),
            #        kulczynski_2(y1, z1),
            #        kulczynski_2(z1, w1))

            #score = (dice_coefficient(x1, y1),
            #        dice_coefficient(x1, z1),
            #        dice_coefficient(x1, w1),
            #        dice_coefficient(x1, y2),
            #        dice_coefficient(y1, z1),
            #        dice_coefficient(z1, w1))

            out[rtype] = np.asarray(score, dtype=np.float)
        return out

    def get_analogy(self, src_feature, target_feature, target_domain, rmax=1, vmax=1):
        """Get the best analogy between two arbitrary features"""

        # ensure features exist
        if not src_feature in self.features:
            print("Feature %s not in source domain" % src_feature)
            return None
        if not target_feature in target_domain.features:
            print("Feature %s not in target domain" % target_feature)
            return None

        tscore = rmax+vmax
        #tscore = 1 
        src_node = self.features[src_feature]
        c_node = target_domain.features[target_feature]

        nc1 = src_node.get_rtype_ratios()
        nc2 = c_node.get_rtype_ratios()


        def get_hypotheses():
            svec = src_node.get_vector2()
            cvec = c_node.get_vector2()
            hypotheses = []

            # precompute source vectors because this won't change
            src_vec_dict = {}
            for r1, d1 in src_node.outgoing_relations:
                d1vec = self.features[d1].get_vector2()
                diff1 = svec - d1vec
                src_vec_dict[(d1, True)] = diff1
            for r1, d1 in src_node.incoming_relations:
                d1vec = self.features[d1].get_vector2()
                diff1 = svec - d1vec
                src_vec_dict[(d1, False)] = diff1

            # for each pair in candidate outgoing
            for r2, d2 in c_node.outgoing_relations:
                d2vec = target_domain.features[d2].get_vector2()
                diff2 = cvec - d2vec
                # find best outgoing rtype to compare with
                for r1, d1 in src_node.outgoing_relations:
                    rdiff = cosine_similarity(self.rtype_index[r1],
                                              target_domain.rtype_index[r2])

                    #weight matches by usage ratio
                    #relatively close usage ratios should have higher confidence
                    rdiff *= 1 - abs(nc1[r1] - nc2[r2])**2



                    diff1 = src_vec_dict[(d1, True)]
                    vdiff = cosine_similarity(diff1, diff2)
                    actual_score = (rdiff*rmax + vdiff*vmax)
                    #actual_score = max(rdiff, vdiff)

                    hypotheses.append((actual_score / tscore, r1, d1, r2, d2, True))
                    #hypotheses.append((rdiff*rmax, r1, d1, r2, d2, True))
                    #hypotheses.append((vdiff*vmax, r1, d1, r2, d2, True))

            # for each pair in candidate incoming
            for r2, d2 in c_node.incoming_relations:
                d2vec = target_domain.features[d2].get_vector2()
                diff2 = cvec - d2vec
                # find best incoming rtype to compare with
                for r1, d1 in src_node.incoming_relations:
                    rdiff = cosine_similarity(self.rtype_index[r1],
                                              target_domain.rtype_index[r2])


                    #weight matches by usage ratio
                    #relatively close usage ratios should have higher confidence
                    rdiff *= 1 - abs(nc1[r1] - nc2[r2])**2

                    diff1 = src_vec_dict[(d1, False)]
                    vdiff = cosine_similarity(diff1, diff2)
                    actual_score = (rdiff*rmax + vdiff*vmax)
                    #actual_score = max(rdiff, vdiff)

                    hypotheses.append((actual_score / tscore, r1, d1, r2, d2, False))
                    #hypotheses.append((rdiff*rmax, r1, d1, r2, d2, False))
                    #hypotheses.append((vdiff*vmax, r1, d1, r2, d2, False))

            hypotheses.sort(reverse=True)
            return hypotheses

        rassert = {}
        hmap = {}
        best = {}
        rating = 0
        total_rating = 0

        # for each mh, pick the best then pick the next best non-conflicting
        for score, r1, src, r2, target, outgoing in get_hypotheses():
            score = score * tscore
            key = (src, outgoing)
            if (hmap.get(key) == target) or (key not in hmap.keys() and\
                                             target not in hmap.values()):
                if r1 != r2 and r1 not in rassert.keys() and\
                        r2 not in rassert.values():
                    if r1 not in c_node.rtypes and\
                       r2 not in src_node.rtypes:  # prevent crossmatching
                        rassert[r1] = r2
                if key not in hmap.keys() and target not in hmap.values():
                    hmap[key] = target
                    total_rating += tscore
                if r1 == r2 or rassert.get(r1) == r2:
                    otype = "OUTGOING" if outgoing else "INCOMING"
                    best[(otype, r1, src)] = (
                        r2, target, score, score / tscore)
                    rating += score
                else:  # penalize inconsistent rtype matchup
                    total_rating += tscore

        # penalize score for non-matches
        #for destobj in src_node.connections:
        #    if (destobj, True) not in hmap.keys():
        #        total_rating += 2

        #for destobj in c_node.connections:
        #    if (destobj, False) not in hmap.values():
        #        total_rating += 2


        #confidence score
        #how confident can you possibly be with an analogy?

        #max numbeer of rtype matches corresponds with
        #max number of total matches

        # number of distinct relationship types
        # number of total relationships

        #score based on relative numbers


        

        tr1 = len(nc1.keys())
        tr2 = len(nc2.keys())

        sr1 = sum(src_node.rtype_count.values())
        sr2 = sum(c_node.rtype_count.values())

        v = max(sr1, sr2)
        z = max(tr1, tr2)

        confidence = 1 - abs(tr1-tr2)/z * abs(sr1-sr2)/v


        #print(target_feature, "confidence: ", confidence)



        #t1 = kulczynski_2(set(rassert.keys()), p1)
        #t2 = kulczynski_2(set(rassert.values()), p2)
        #weight = (t1 + t2) / 2



        if total_rating == 0:  # prevent divide by zero error
            return None

        normalized_rating = rating / total_rating

        #normalized_rating *= confidence
        total_score = confidence * normalized_rating #(confidence/3 + normalized_rating*2/3 + weight/3)

        #return (normalized_rating, rating, total_rating,
        #        (src_feature, target_feature), rassert, best)

        return {"total_score":total_score,
                "confidence":confidence,
                "rating":normalized_rating,
                "src":src_feature,
                "target":target_feature,
                "asserts":rassert,
                "mapping":list(best.items()), #have to be json friendly
                "explanation":self.explain_analogy(src_feature, target_feature, best)}

        #return (total_score, normalized_rating, confidence, target_feature)

    def find_best_analogy(self, src_feature, target_domain, filter_list=None, rmax=1, vmax=1):
        """
        Finds the best analogy between a specific feature in the source domain
        and any feature in the target domain.

        If filter_list is specified, only the features in that list will be
        selected from to make analogies.

        Note: analogies to self are ignored (if same domain)
        """

        candidate_pool = filter_list if filter_list is not None else target_domain.features
        candidate_results = []

        for c_feature in candidate_pool:
            # find novel within same domain
            if target_domain == self and c_feature == src_feature:
                continue
            result = self.get_analogy(src_feature, c_feature, target_domain, rmax, vmax)
            if result:
                candidate_results.append(result)

        if not candidate_results:
            return None
        else:
            #return best global analogy
            return sorted(candidate_results, key=lambda x: x["total_score"])[-1]

    def get_all_analogies(self, src_feature, target_domain, filter_list=None):
        """
        Returns all analogies between a specific feature in the source domain
        and all features in the target domain.

        If filter_list is specified, only the features in that list will be
        selected from to make analogies.

        """

        candidate_pool = filter_list if filter_list is not None else target_domain.features
        results = []

        for target_feature in candidate_pool:
            result = self.get_analogy(
                src_feature, target_feature, target_domain)
            if result:
                results.append(result)
        return results
