"""
analogy2.py

module for making analogies between standard graph structures


"""

import numpy as np
from utils import permute_rtype_vector, cosine_similarity, euclidean_distance, NULL_VEC
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

def make_analogy(src_concept, src_domain, target_concept, target_domain, rmax=1, vmax=1, relaxed=False):
    '''Makes the best analogy between two concepts in two domains

    src_domain is the KNOWN domain
    target_domain is the NOVEL domain

    if relaxed is set to True, the analogy evidence will not be one-to-one

    returns the best analogy or None if no analogy could be made

    '''
    
    

    # ensure features exist
    if not src_concept in src_domain.nodes:
        print("Feature %s not in source domain" % src_concept)
        return None
    if not target_concept in target_domain.nodes:
        print("Feature %s not in target domain" % target_concept)
        return None

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

        #return 1 - (nc1[r1] - nc2[r2])**2

    def weigh_score(x,c):
        #return math.exp((3*x-3))
        #return math.tanh((6*x-6) + 2)
        return math.tanh(2*c*x - c)

    def get_hypotheses():
        svec = src_domain.node_vectors[src_concept]
        tvec = target_domain.node_vectors[target_concept]

        hypotheses = []

        # precompute source vectors because this won't change
        src_vec_dict = {}

        
        net1 = [(r1,d1,True) for r1,d1 in cnode.outgoing_relations] +\
               [(r1,d1,False) for r1,d1 in cnode.incoming_relations]

        net2 = [(r1,d1,True) for r1,d1 in tnode.outgoing_relations] +\
               [(r1,d1,False) for r1,d1 in tnode.incoming_relations]

        #precompute vectors for inner loop
        for r, d, v in net1:
            #vector from src node to src neighbor
            src_vec_dict[d] = svec - src_domain.node_vectors[d]

            #vector from src node to src rtype
            if v: #if outgoing rtype
                src_vec_dict[(r,v)] = svec - src_domain.rtype_vectors[r]
            else: #if incoming rtype
                src_vec_dict[(r,v)] = svec - permute_rtype_vector(src_domain.rtype_vectors[r])


        # for each pair in candidate outgoing
        for r2, d2, v2 in net2:

            #vector from target node to target neighbor
            vdiff2 = tvec - target_domain.node_vectors[d2]

            #vector from target node to target rtype
            if v2: #if outgoing rtype
                rdiff2 = tvec - target_domain.rtype_vectors[r2]
            else: #if incoming rtype
                rdiff2 = tvec - permute_rtype_vector(target_domain.rtype_vectors[r2])

            # find best outgoing rtype to compare with
            for r1, d1, v1 in net1:

                #compute relative rtype score
                rscore = cosine_similarity(src_vec_dict[(r1,v1)], rdiff2)

                #adjust rtype score by confidence
                rscore *= get_confidence(r1,r2)

                #skew score
                rscore = math.tanh(2*math.e*rscore - math.e)

                #compute relative node score
                vscore = cosine_similarity(src_vec_dict[d1], vdiff2)

                actual_score = (rscore*rmax + vscore*vmax)/tscore

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
    ritems = rassert.items()

    # for each mh, pick the best then pick the next best non-conflicting
    # total number of hypotheses is O(m*n) with max(n,m) matches
    # total score is based on possible non-conflicts
    # only penalize on conflicts

    for score, r1, src, r2, target, v1, v2 in get_hypotheses():
        vkey = (src, v1)
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

        #for a new mapping
        if vkey not in hkeys and (relaxed or (target not in hvals)):
            
            #print("new mapping: ", vkey, target)
            hmap[vkey] = target
            rating += score
            if (r1,r2) not in ritems:
                total_rating += tscore
            if rkey1 not in rkeys and rkey2 not in rvals:
                rassert[rkey1] = rkey2

        if target not in hvals:
            total_rating += tscore

        #if the src/target has already been mapped to
        #if hmap.get(vkey) == target:
        if hmap.get(vkey) == target:# or target not in hvals:
            #check for conflict with relationship types
            if rassert.get(rkey1) == rkey2:
                #track best match
                best[(otype, r1, src)] = (r2, target, score)
                #increase score for match
                rating += score
    
    #total number of rtypes for each node
    tr1 = len(nc1.keys())
    tr2 = len(nc2.keys())

    #total number of relationships for each node
    sr1 = sum(cnode.rtype_count.values())
    sr2 = sum(tnode.rtype_count.values())

    #get max number of each (ideal number)
    v = max(sr1, sr2)
    z = max(tr1, tr2)

    weight = z*v

    #confidence is based on difference from ideal
    confidence = 1 - (abs(tr1-tr2)/z + abs(sr1-sr2)/v)/2

    if total_rating == 0:  # prevent divide by zero error
        return None

    normalized_rating = rating / total_rating #* math.log(weight)

    total_score = confidence * normalized_rating

    return {"total_score":total_score,
            "confidence":confidence,
            "rating":normalized_rating,
            "src_concept":src_concept,
            "target_concept":target_concept,
            "asserts":rassert,
            "mapping":best,
            "weight":weight}


def find_best_analogy(src_concept, src_domain, target_domain, filter_list=None, rmax=1, vmax=1):
    """Makes the best analogy between two concepts in two domains

    Finds the best analogy between a specific concept in the source domain
    and any concept in the target domain.

    If filter_list is specified, only the concepts in that list will be
    selected from to make analogies.

    Note: analogies to self are ignored (if same domain)
    """

    candidate_pool = filter_list if filter_list is not None else target_domain.nodes
    candidate_results = []

    for target_concept in candidate_pool:
        # find novel within same domain
        # otherwise best analogy would always be self
        if target_domain == src_domain and target_concept == src_concept:
            continue
        result = make_analogy(src_concept, src_domain, target_concept, target_domain, rmax, vmax)
        if result:
            candidate_results.append(result)

    if not candidate_results:
        return None
    else:
        
        
        tmp = sorted(candidate_results, key=lambda x: x["total_score"])
        #from pprint import pprint
        #tmp2 = [(x["total_score"],x["confidence"],x["rating"],x["target_concept"]) for x in tmp]
        #pprint(tmp2)

        # return the best global analogy
        return tmp[-1]


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