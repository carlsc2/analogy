import xml.etree.ElementTree as ET
from collections import Counter
from math import sqrt
from pprint import pprint
import numpy as np
from lru import LRU


JACCARD_DIMENSIONS = 28
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

        self.knowledge_level = len(
            self.outgoing_relations) + len(self.incoming_relations)

        self._vector = None
        self._vector2 = None

        self.text = ""

    def add_predecessor(self, rtype, pred):
        self.incoming_relations.add((rtype, pred))
        self.predecessors.add(pred)
        self.knowledge_level = len(
            self.outgoing_relations) + len(self.incoming_relations)
        self._vector = None
        self._vector2 = None

    def add_relation(self, rtype, dest):
        self.connections.add(dest)
        self.outgoing_relations.add((rtype, dest))
        self.rtypes.add(rtype)
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
            #a = np.asarray(tmp1).mean(axis=0)
            #b = np.asarray(tmp2).mean(axis=0)
            #c = np.empty((a.size + b.size,), dtype=a.dtype)
            #c[0::2] = a
            #c[1::2] = b
            #self._vector = c
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

    def explain_analogy(self, analogy, verbose=False):
        # only explain main relation
        if not analogy:
            return

        nrating, rating, total_rating, (src, trg), rassert, mapping = analogy

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
                    hm[rtype] = (Counter(), Counter(), Counter(), Counter(),
                                 Counter(), Counter(), Counter(), Counter())
                lco, gco, smo, dfo, lci, gci, smi, dfi = hm[rtype]

                for r in loses:
                    lco[r] += 1
                for r in gains:
                    gco[r] += 1
                for r in same:
                    smo[r] += 1
                for r in diff:
                    dfo[r] += 1

            for (rtype, src) in fnode.incoming_relations:
                loses = fnode.rtypes - self.features[src].rtypes
                gains = self.features[src].rtypes - fnode.rtypes
                same = self.features[src].rtypes & fnode.rtypes
                diff = self.features[src].rtypes ^ fnode.rtypes

                if rtype not in hm:
                    hm[rtype] = (Counter(), Counter(), Counter(), Counter(),
                                 Counter(), Counter(), Counter(), Counter())
                lco, gco, smo, dfo, lci, gci, smi, dfi = hm[rtype]

                for r in loses:
                    lci[r] += 1
                for r in gains:
                    gci[r] += 1
                for r in same:
                    smi[r] += 1
                for r in diff:
                    dfi[r] += 1

        out = {}  # compute metrics from rtypes
        for rtype, (lco, gco, smo, dfo, lci, gci, smi, dfi) in hm.items():
            x1 = set(lco)
            y1 = set(gco)
            z1 = set(smo)
            w1 = set(dfo)

            x2 = set(lci)
            y2 = set(gci)
            z2 = set(smi)
            w2 = set(dfi)

            #def adjust(x):  # eliminate outlier data for better results
            #    total_count = sum(x.values())
            #    return set(a for a, b in x.items() if b / total_count > .25)

            #x1 = adjust(lco)
            #y1 = adjust(gco)
            #z1 = adjust(smo)
            #w1 = adjust(dfo)

            #x2 = adjust(lci)
            #y2 = adjust(gci)
            #z2 = adjust(smi)
            #w2 = adjust(dfi)

            score = (jaccard_index(x1, y1),
                     jaccard_index(x1, z1),
                     jaccard_index(x1, w1),
                     jaccard_index(x1, x2),
                     jaccard_index(x1, y2),
                     jaccard_index(x1, z2),
                     jaccard_index(x1, w2),
                     jaccard_index(y1, z1),
                     jaccard_index(y1, w1),
                     jaccard_index(y1, x2),
                     jaccard_index(y1, y2),
                     jaccard_index(y1, z2),
                     jaccard_index(y1, w2),
                     jaccard_index(z1, w1),
                     jaccard_index(z1, x2),
                     jaccard_index(z1, y2),
                     jaccard_index(z1, z2),
                     jaccard_index(z1, w2),
                     jaccard_index(w1, x2),
                     jaccard_index(w1, y2),
                     jaccard_index(w1, z2),
                     jaccard_index(w1, w2),
                     jaccard_index(x2, y2),
                     jaccard_index(x2, z2),
                     jaccard_index(x2, w2),
                     jaccard_index(y2, z2),
                     jaccard_index(y2, w2),
                     jaccard_index(z2, w2))

            out[rtype] = np.asarray(score, dtype=np.float)
        return out

    def get_analogy(self, src_feature, target_feature, target_domain, assertions=None, rmax=1, vmax=1):
        """Get the best analogy between two arbitrary features"""

        # ensure features exist
        if not src_feature in self.features:
            print("Feature %s not in source domain" % src_feature)
            return None
        if not target_feature in target_domain.features:
            print("Feature %s not in target domain" % target_feature)
            return None

        if not assertions:
            assertions = {}

        src_node = self.features[src_feature]
        target_node = target_domain.features[target_feature]

        def get_hypotheses(src_node,target_node):
            svec = src_node.get_vector2()
            cvec = target_node.get_vector2()
            hypotheses = Counter()
            h2 = Counter()

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
            for r2, d2 in target_node.outgoing_relations:
                d2vec = target_domain.features[d2].get_vector2()
                diff2 = cvec - d2vec
                # find best outgoing rtype to compare with
                for r1, d1 in src_node.outgoing_relations:

                    if d1 != d2 and (d1 in assertions.keys() or d2 in assertions.values()):
                        if assertions.get(d1) != d2:
                            continue

                    if r1 != r2 and (r1 in assertions.keys() or r2 in assertions.values()):
                        if assertions.get(r1) != r2:
                            continue

                    rdiff = cosine_similarity(self.rtype_index[r1],
                                                target_domain.rtype_index[r2])
                    diff1 = src_vec_dict[(d1, True)]
                    vdiff = cosine_similarity(diff1, diff2)

                    hypotheses[(r1, r2)] += rdiff*rmax
                    h2[(r1, r2)] += 1
                    hypotheses[(d1, d2)] += vdiff*vmax
                    h2[(d1, d2)] += 1

            # for each pair in candidate incoming
            for r2, d2 in target_node.incoming_relations:
                d2vec = target_domain.features[d2].get_vector2()
                diff2 = cvec - d2vec
                # find best incoming rtype to compare with
                for r1, d1 in src_node.incoming_relations:

                    if d1 != d2 and (d1 in assertions.keys() or d2 in assertions.values()):
                        if assertions.get(d1) != d2:
                            continue

                    if r1 != r2 and (r1 in assertions.keys() or r2 in assertions.values()):
                        if assertions.get(r1) != r2:
                            continue



                    rdiff = cosine_similarity(self.rtype_index[r1],
                                                target_domain.rtype_index[r2])
                    diff1 = src_vec_dict[(d1, False)]
                    vdiff = cosine_similarity(diff1, diff2)

                    hypotheses[(r2, r1)] += rdiff*rmax
                    h2[(r2, r1)] += 1
                    hypotheses[(d2, d1)] += vdiff*vmax
                    h2[(d2, d1)] += 1

            return hypotheses,h2

        return get_hypotheses(src_node,target_node)

    def find_best_analogy(self, src_feature, target_domain, filter_list=None, rmax=1, vmax=1):
        """
        Finds the best analogy between a specific feature in the source domain
        and any feature in the target domain.

        If filter_list is specified, only the features in that list will be
        selected from to make analogies.

        Note: analogies to self are ignored (if same domain)
        """

        candidate_pool = filter_list if filter_list is not None else target_domain.features

        net_hypos = Counter()
        net_counts = Counter()


        for c_feature in candidate_pool:
            # find novel within same domain
            if target_domain == self and c_feature == src_feature:
                continue
            result = self.get_analogy(src_feature, c_feature, target_domain, None, rmax, vmax)
            if result:
                net_hypos += result[0]
                net_counts += result[1]

        ret =  Counter({a:b / net_counts[a] for a, b in net_hypos.items()})

        asserts = {}
        for (v1,v2),b in ret.most_common():
            if v1 not in asserts.keys() and v2 not in asserts.values():
                asserts[v1] = v2


        for src_feature in asserts.keys():
            for target_feature in asserts.values():
                result = self.get_analogy(src_feature, target_feature, target_domain, asserts, rmax, vmax)
                if result:
                    net_hypos += result[0]
                    net_counts += result[1]
                else:
                    result = self.get_analogy(target_feature, src_feature, target_domain, asserts, rmax, vmax)
                    if result:
                        net_hypos += result[0]
                        net_counts += result[1]

        ret =  Counter({a:b / net_counts[a] for a, b in net_hypos.items()})

        asserts2 = {}
        for (v1,v2),b in ret.most_common():
            if v1 not in asserts2.keys() and v2 not in asserts2.values():
                asserts2[v1] = v2

        return asserts2

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
