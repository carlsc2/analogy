"""
analogy2.py

module for making analogies between standard graph structures


"""

import numpy as np
from utils import permute_rtype_vector, cosine_similarity



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

def make_analogy(src_concept, src_domain, target_concept, target_domain, rmax=1, vmax=1, threshold=0.5):
    '''Makes the best analogy between two concepts in two domains

    src_domain is the KNOWN domain
    target_domain is the NOVEL domain

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

    tscore = rmax + vmax

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
            src_vec_dict[d] = svec - src_domain.node_vectors[d]

            if v: #if outgoing rtype
                src_vec_dict[(r,v)] = svec - src_domain.rtype_vectors[r]
            else: #if incoming rtype
                src_vec_dict[(r,v)] = svec - permute_rtype_vector(src_domain.rtype_vectors[r])


        # for each pair in candidate outgoing
        for r2, d2, v2 in net2:

            vdiff2 = tvec - target_domain.node_vectors[d2]

            if v2: #if outgoing rtype
                rdiff2 = tvec - target_domain.rtype_vectors[r2]
            else: #if incoming rtype
                rdiff2 = tvec - permute_rtype_vector(target_domain.rtype_vectors[r2])

            # find best outgoing rtype to compare with
            for r1, d1, v1 in net1:


                #compute relative rtype score
                rscore = cosine_similarity(src_vec_dict[(r1,v1)], rdiff2)

                #adjust rtype score by confidence
                rscore *= 1 - abs(nc1[r1] - nc2[r2])**2

                #compute node score
                vscore = cosine_similarity(src_vec_dict[d1], vdiff2)

                actual_score = (rscore*rmax + vscore*vmax)

                if actual_score >= threshold:
                    hypotheses.append((actual_score / tscore, r1, d1, r2, d2, v1, v2))

        hypotheses.sort(reverse=True)
        return hypotheses

    rassert = {}
    hmap = {}
    best = {}
    rating = 0
    total_rating = 0

    # for each mh, pick the best then pick the next best non-conflicting
    for score, r1, src, r2, target, v1, v2 in get_hypotheses():
        #score = score * tscore
        key = (src, v1)
        if (hmap.get(key) == target) or (key not in hmap.keys() and\
                                            target not in hmap.values()):
            if r1 != r2 and r1 not in rassert.keys() and\
                    r2 not in rassert.values():
                #if r1 not in tnode.rtypes and\
                #    r2 not in cnode.rtypes:  # prevent crossmatching
                    rassert[r1] = r2
            if key not in hmap.keys() and target not in hmap.values():
                hmap[key] = target
                total_rating += tscore
            if r1 == r2 or rassert.get(r1) == r2:
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

                best[(otype, r1, src)] = (r2, target, score)
                rating += score
            else:  # penalize inconsistent rtype matchup
                total_rating += tscore      

    tr1 = len(nc1.keys())
    tr2 = len(nc2.keys())

    sr1 = sum(cnode.rtype_count.values())
    sr2 = sum(tnode.rtype_count.values())

    v = max(sr1, sr2)
    z = max(tr1, tr2)

    confidence = 1 - abs(tr1-tr2)/z * abs(sr1-sr2)/v

    if total_rating == 0:  # prevent divide by zero error
        return None

    normalized_rating = rating #/ total_rating

    total_score = confidence * normalized_rating

    return {"total_score":total_score,
            "confidence":confidence,
            "rating":normalized_rating,
            "src_concept":src_concept,
            "target_concept":target_concept,
            "asserts":rassert,
            "mapping":best}


def find_best_analogy(src_concept, src_domain, target_domain, filter_list=None, rmax=1, vmax=1, threshold=1):
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
        result = make_analogy(src_concept, src_domain, target_concept, target_domain, rmax, vmax, threshold)
        if result:
            candidate_results.append(result)

    if not candidate_results:
        return None
    else:
        
        
        tmp = sorted(candidate_results, key=lambda x: x["total_score"])
        from pprint import pprint
        tmp2 = [(x["total_score"],x["confidence"],x["rating"],x["target_concept"]) for x in tmp]
        pprint(tmp2)

        # return the best global analogy
        return tmp[-1]


