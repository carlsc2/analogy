"""
analogy2.py

module for making analogies between standard graph structures


"""

import numpy as np
from utils import permute_rtype_vector, cosine_similarity, euclidean_distance
import math

from pprint import pprint

'''
To make analogies:
- need to start out with initial assertions
    - these assertions are either known to be good or are based on intent of analogy

Types of analogies:
1) If A is novel and B is known: Explain A in terms of B
2) If neither A nor B is understood: Relate A to B and explain in terms of C,
    where C is understood

Need to incrementally make inferences
- if X is like Y then X-->A is like Y-->B


'''

class AnalogyException(Exception):
    pass

def make_analogy(src_concept, src_domain, target_concept, target_domain, rmax=1, vmax=1, singular=False):
    '''Makes the best analogy between two concepts in two domains

    src_domain is the KNOWN domain
    target_domain is the NOVEL domain

    returns the best analogy that can be made between the two concepts

    if singular is True, the analogy will be computed using a single example of 
    each relationship type, weighted appropriately. This is useful for nodes with
    a very large number of homogeneous connections as it severely cuts down on
    the computation time.

    raises an AnalogyException if concept does not exist in domain 
    '''
    
    

    # ensure features exist
    if not src_concept in src_domain.nodes:
        raise AnalogyException("'%s' not in source domain" % src_concept)

    if not target_concept in target_domain.nodes:
        raise AnalogyException("'%s' not in target domain" % target_concept)

    cnode = src_domain.nodes[src_concept]
    tnode = target_domain.nodes[target_concept]

    nc1 = cnode.get_rtype_ratios()
    nc2 = tnode.get_rtype_ratios()

    tv1 = sum([len(x) for x in src_domain.usage_map.values()])
    tv2 = sum([len(x) for x in target_domain.usage_map.values()])

    tscore = rmax + vmax

    def get_confidence(r1,r2):
        #get the relative confidence between two relationship types

        ##confidence based on total knowledge
        c1 = len(src_domain.usage_map[r1])
        c2 = len(target_domain.usage_map[r2])

        ratio1 = c1/tv1
        ratio2 = c2/tv2

        diff1 = (ratio1 - ratio2)**2

        #confidence based on relative usage
        diff2 = (nc1[r1] - nc2[r2])**2

        return 1 - (diff1+diff2)/2

    def get_hypotheses():
        svec = src_domain.node_vectors[src_concept]
        tvec = target_domain.node_vectors[target_concept]

        hypotheses = []

        # precompute source vectors because this won't change
        src_vec_dict = {}

        #aggregate incominng/outgoing
        net2 = [(r,d,True) for r,d in tnode.outgoing_relations] +\
                [(r,d,False) for r,d in tnode.incoming_relations]

        #only use one of each rtype
        if singular:
            for rtype in cnode.rtypes:
                clv = np.mean([src_domain.node_vectors[d] for r,d \
                    in cnode.outgoing_relations if r == rtype], axis=0)
                src_vec_dict[(rtype,
                              "things with %s from"%rtype,
                              True)] = (svec - clv,
                                        svec - src_domain.rtype_vectors[rtype])

            for rtype in cnode.i_rtypes:
                clv = np.mean([src_domain.node_vectors[d] for r,d \
                    in cnode.incoming_relations if r == rtype], axis=0)
                src_vec_dict[(rtype,
                              "things with %s to"%rtype,
                              False)] = (svec - clv,
                                         svec - permute_rtype_vector(
                                             src_domain.rtype_vectors[rtype]))

        else:
            #precompute vectors for inner loop
            for r,d in cnode.outgoing_relations:
                #vector from src node to src neighbor, vector from src node to src rtype
                src_vec_dict[(r,d,True)] = (svec - src_domain.node_vectors[d],
                                            svec - src_domain.rtype_vectors[r])
            for r,d in cnode.incoming_relations:
                src_vec_dict[(r,d,False)] = (svec - src_domain.node_vectors[d],
                                             svec - permute_rtype_vector(
                                                 src_domain.rtype_vectors[r]))

        svdi = src_vec_dict.items()

        # for each pair in target in/out
        for r2, d2, v2 in net2:

            #vector from target node to target neighbor
            vdiff2 = tvec - target_domain.node_vectors[d2]

            #vector from target node to target rtype
            if v2: #if outgoing rtype
                rdiff2 = tvec - target_domain.rtype_vectors[r2]
            else: #if incoming rtype
                rdiff2 = tvec - permute_rtype_vector(target_domain.rtype_vectors[r2])

            #compare with each pair in source in/out
            for (r1, d1, v1), (vdiff1, rdiff1) in svdi:

                #compute relative rtype score
                rscore = cosine_similarity(rdiff1, rdiff2)

                #adjust rtype score by confidence
                rscore *= get_confidence(r1,r2)

                #skew score
                #rscore = math.tanh(2*math.e*rscore - math.e)

                #compute relative node score
                vscore = cosine_similarity(vdiff1, vdiff2)

                actual_score = (rscore*rmax + vscore*vmax)/tscore

                if singular:
                    actual_score *= nc1[r1]

                hypotheses.append((actual_score, r1, d1, r2, d2, v1, v2))

        hypotheses.sort(reverse=True)
        return hypotheses

    rassert = {}
    hmap = {}
    best = {}
    rating = 0
    total_rating = 0

    #use dict views to avoid copying
    hkeys = hmap.keys()
    hvals = hmap.values()
    rkeys = rassert.keys()
    rvals = rassert.values()

    #total number of rtypes for each node
    tr1 = len(nc1.keys())
    tr2 = len(nc2.keys())

    #total number of relationships for each node
    sr1 = len(cnode.outgoing_relations) + len(cnode.incoming_relations)
    sr2 = len(tnode.outgoing_relations) + len(tnode.incoming_relations)

    # for each mh, pick the best then pick the next best non-conflicting
    # total number of hypotheses is O(m*n) with max(n,m) matches
    # total score is based on possible non-conflicts
    # only penalize on true conflicts
    # max possible matches is min(n,m) matches
    maxm = min(sr1,sr2)
    #total_rating = maxm

    for score, r1, src, r2, target, v1, v2 in get_hypotheses():
        vkey = (src, v1)
        tkey = (target, v2)
        rkey1 = (r1, v1)
        rkey2 = (r2, v2)

        if v1:
            if v2:
                otype = "OUT-OUT"
            else:
                otype = "OUT-IN"
        else:
            if v2:
                otype = "IN-OUT"
            else:
                otype = "IN-IN"

        #same src, dest could have multiple rtype mappings

        #if new concept mapping
        if vkey not in hkeys and tkey not in hvals:
            #if new rtype mapping
            if rkey1 not in rkeys and rkey2 not in rvals:
                rassert[rkey1] = rkey2
                hmap[vkey] = tkey
                best[(otype, r1, src)] = (r2, target, score)
                rating += score
            #if rtype mapping exists but is consistent
            elif rassert.get(rkey1) == rkey2:
                hmap[vkey] = tkey
                best[(otype, r1, src)] = (r2, target, score)
                rating += score
        #if existing concept mapping
        elif hmap.get(vkey) == tkey:
            #if new rtype mapping
            if rkey1 not in rkeys and rkey2 not in rvals:
                rassert[rkey1] = rkey2
                hmap[vkey] = tkey
                best[(otype, r1, src)] = (r2, target, score)
                rating += score
            #if rtype mapping exists but is consistent
            elif rassert.get(rkey1) == rkey2:
                hmap[vkey] = tkey
                best[(otype, r1, src)] = (r2, target, score)
                rating += score
        total_rating += 1/maxm

    #get max number of each (ideal number)
    v = max(sr1, sr2)
    z = max(tr1, tr2)

    #total impact of the analogy (in terms of connectivity)
    weight = z*v

    #confidence is based on difference from ideal
    #best possible analogy:
    #   all types could be mapped, 
    #   all connections could be mapped 

    mass = len(hmap) * len(rassert) / weight

    confidence = 1 - (abs(tr1-tr2)/z + abs(sr1-sr2)/v)/2

    if total_rating != 0:  # prevent divide by zero error
        normalized_rating = rating / total_rating
    else:
         normalized_rating = 0

    #total_score = ((confidence *2) + normalized_rating) / 3 #weigh confidence more
    total_score = (confidence + normalized_rating) / 2

    return {"total_score":total_score,
            "confidence":confidence,
            "rating":normalized_rating,
            "src_concept":src_concept,
            "target_concept":target_concept,
            "asserts":rassert,
            "mapping":best,
            "weight":weight}

def find_best_analogy(src_concept, src_domain, target_domain, filter_list=None, rmax=1, vmax=1, singular=False):
    """Makes the best analogy between two concepts in two domains

    Finds the best analogy between a specific concept in the source domain
    and any concept in the target domain.

    If filter_list is specified, only the concepts in that list will be
    selected from to make analogies.

    Note: analogies to self are ignored (if same domain)

    raises an AnalogyException if concept does not exist in domain 
    """
    candidate_pool = filter_list if filter_list is not None else target_domain.nodes

    if not src_concept in src_domain.nodes:
        raise AnalogyException("'%s' not in source domain" % src_concept)

    best_result = None
    best_score = 0

    for target_concept in candidate_pool:
        # find novel within same domain
        # otherwise best analogy would always be self
        if target_domain == src_domain and target_concept == src_concept:
            continue
        result = make_analogy(src_concept, src_domain, target_concept, target_domain, rmax, vmax, singular)
        if result["total_score"] > best_score:
            best_result = result
            best_score = result["total_score"]

    return best_result


def get_all_analogies(src_concept, src_domain, target_domain, filter_list=None, rmax=1, vmax=1, singular=False):
    """Makes all analogies for some concept in one domain to another domain

    Finds all analogies between a specific concept in the source domain
    and any concept in the target domain.

    If filter_list is specified, only the concepts in that list will be
    selected from to make analogies.

    raises an AnalogyException if concept does not exist in domain 
    """

    candidate_pool = filter_list if filter_list is not None else target_domain.nodes
    results = []

    if not src_concept in src_domain.nodes:
        raise AnalogyException("'%s' not in source domain" % src_concept)

    for target_concept in candidate_pool:
        result = make_analogy(src_concept, src_domain, target_concept, target_domain, rmax, vmax, singular)
        if result:
            results.append(result)     
             
    return results


def explain_analogy(analogy, verbose=False):
    # only explain main relation
    if not analogy:
        return

    src = analogy["src_concept"]
    trg = analogy["target_concept"]
    mapping = analogy["mapping"]


    narrative = ""
    narrative += "\t%s is like %s. " % (src, trg)

    narrative += "This is because"
    nchunks = []

    mentioned = set()

    for (x, r1, d1), (r2, d2, s) in mapping.items():
        if not verbose and r1 in mentioned:
            continue
        if x == "IN-IN":
            nchunks.append((s, d1, r1, src, d2, r2, trg))
        if x == "IN-OUT":
            nchunks.append((s, d1, r1, src, trg, r2, d2))
        if x == "OUT-IN":
            nchunks.append((s, src, r1, d1, d2, r2, trg))
        if x == "OUT-OUT":
            nchunks.append((s, src, r1, d1, trg, r2, d2))
        mentioned.add(r1)
    nchunks.sort(reverse=True)
    #order by score to give most important matches first 
    for i, nc in enumerate(nchunks):
        s, a, b, c, d, e, f = nc
        if i == len(nchunks) - 1:
            if i > 1: #only add and if more than one thing
                narrative += " and"
            narrative += " '%s' <%s> '%s' in the same way that '%s' <%s> '%s'.\n" % (
                a, b, c, d, e, f)
        else:
            narrative += " '%s' <%s> '%s' in the same way that '%s' <%s> '%s'," % (
                a, b, c, d, e, f)
    return narrative