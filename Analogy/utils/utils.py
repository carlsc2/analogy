"""
utils.py

Contains definitions for graph structures and basic operations on those
structures.

"""

import xml.etree.ElementTree as ET
from collections import Counter
from math import sqrt
import numpy as np
from lru import LRU
from scipy.spatial.ckdtree import cKDTree
import numpy as np
import json
import pickle
import os.path
from math import log

def softmax(x):
    """Compute softmax values for x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
    la = len(a)
    lb = len(b)
    if la == lb == 0:  # if both sets are empty, return 1
        return 1
    n = len(a&b)
    return n / (la + lb - n)


def dice_coefficient(a, b):
    '''Computes the dice coefficient between two sets

    1 means completely similar, 0 means completely different.'''
    total = (len(a) + len(b))
    if total == 0:
        return 1
    overlap = len(a & b)
    return overlap * 2.0 / total

def permute_rtype_vector(x):
    """convert incoming relationship to outgoing and vice versa"""

    return np.array([x[0],x[5],x[6],x[7],x[8],x[1],x[2],
                     x[3],x[4],x[9],x[11],x[10],x[12]],dtype=np.float)

    #return np.array([x[0],x[5],x[6],x[7],x[8],x[1],x[2],
    #                 x[3],x[4],x[9],x[13],x[14],x[15],x[10],
    #                 x[11],x[12],x[16],x[17],x[22],x[23],x[24],
    #                 x[25],x[18],x[19],x[20],x[21],x[27],x[26],x[28]],
    #                dtype=np.float)


def PCA(data, n=2):
    """Perform Principal Component Analysis on a matrix

    Returns the projection of the data onto the first <n> principal components.

    """
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    s = np.diag(S)
    newdata = np.dot(U[:, :n], np.dot(s[:n, :n], Vt[:n,:]))
    return newdata


class ConsolidatorException(Exception):
    """Raise this exception in a consolidator if some rtype should be ignored"""
    pass



#JACCARD_DIMENSIONS = 29
JACCARD_DIMENSIONS = 13
NULL_VEC = lambda : np.zeros(JACCARD_DIMENSIONS)

SIMILARITY_CACHE = LRU(10000) #use LRU cache to limit size and avoid memory error


def euclidean_distance(v1, v2, _f=np.sum):
    '''Computes the euclidean distance between two vectors'''
    return sqrt(_f((v1 - v2)**2))


def cosine_similarity(v1, v2):
    '''Computes the cosine similarity between two vectors
    
    Result is a value between 0 and 1, with 1 being most similar
    
    '''
    key = (v1.data.tobytes(), v2.data.tobytes())
    try:
        return SIMILARITY_CACHE[key]
    except KeyError:
        nu = v1.dot(v1)
        nv = v2.dot(v2)
        if nu == 0 or nv == 0:
            if nv == 0 and nu == 0:
                value = 1
            else:
                value = 0
        else:
            value = 0.5 * (v1.dot(v2) / sqrt(nu * nv) + 1)

        SIMILARITY_CACHE[key] = value
        return value



class Node:
    '''Represents a node in a graph
    
    Contains all relevant graph information for a particular entity.
    
    '''

    def __init__(self, name):
        self.name = name
        self.attributes = set() #set of (attribute, literal value) pairs
        self.outgoing_relations = set()  # set of relations to other nodes
        self.incoming_relations = set()  # set of relations from other nodes
        self.rtypes = set() #set of types of outgoing relationships
        self.i_rtypes = set() #set of types of incoming relationships
        self.atypes = set() #set of types of attributes
        self.rtype_count = Counter() #how many times each rtype is used
        self.text = ""

        #store computed data for a particular domain
        self._vec_dict = None
        self._cluster_vec_dict = None
        self.domain = None


    @property
    def knowledge_level(self):
        return len(self.outgoing_relations) +\
               len(self.incoming_relations) #+\
               #len(self.attributes)

    def get_rtype_ratios(self):
        total = sum(self.rtype_count.values())
        return {x:self.rtype_count[x]/total for x in self.rtype_count}

    def get_vec_dict(self, domain, cluster=False):
        #return vector dict if computed with same parameters
        if domain != self.domain:
            self.compute_dicts(domain)

        if cluster:
            return self._cluster_vec_dict
        else:
            return self._vec_dict

    def compute_dicts(self, domain):
        #recompute vector dicts
     
        vec_dict = {}
        cluster_vec_dict = {}
        svec = domain.node_vectors[self.name]

        #compute clustered dict values
        for rtype in self.rtypes:
            cnds = [domain.node_vectors[d] for r,d
                    in self.outgoing_relations if r == rtype]
            clen = len(cnds)
            if clen > 1:
                cluster_vec_dict[(rtype, "things are <%s> from"%rtype, True)] = (
                    svec - np.mean(cnds, axis=0),
                    svec - domain.rtype_vectors[rtype])
                    #domain.rtype_vectors[rtype])
            elif clen == 1:
                d = next(d for r,d in self.outgoing_relations if r == rtype)
                cluster_vec_dict[(rtype, "%s <%s> of"%(d, rtype), True)] = (
                    svec - domain.node_vectors[d],
                    svec - domain.rtype_vectors[rtype])
                    #domain.rtype_vectors[rtype])

        for rtype in self.i_rtypes:
            cnds = [domain.node_vectors[d] for r,d
                    in self.incoming_relations if r == rtype]
            clen = len(cnds)
            if clen > 1:
                cluster_vec_dict[(rtype, "things are <%s> to"%rtype, False)] = (
                    svec - np.mean(cnds, axis=0),
                    svec - permute_rtype_vector(domain.rtype_vectors[rtype]))
                    #permute_rtype_vector(domain.rtype_vectors[rtype]))
            elif clen == 1:
                d = next(d for r,d in self.incoming_relations if r == rtype)
                cluster_vec_dict[(rtype, "%s <%s>"%(d, rtype), False)] = (
                    svec - domain.node_vectors[d],
                    svec - permute_rtype_vector(domain.rtype_vectors[rtype]))
                    #permute_rtype_vector(domain.rtype_vectors[rtype]))

        #compute individual dict values
        vec_dict = {(r,d,False):(svec - domain.node_vectors[d],
                                 svec - permute_rtype_vector(
                                        domain.rtype_vectors[r]))
                                    #permute_rtype_vector(
                                    #    domain.rtype_vectors[r]))
                        for r,d in self.incoming_relations}

        for r,d in self.outgoing_relations:
            #vector from node to neighbor, vector from node to rtype
            vec_dict[(r,d,True)] = (svec - domain.node_vectors[d],
                                    svec - domain.rtype_vectors[r])
                                    #domain.rtype_vectors[r])

        self._vec_dict = vec_dict
        self._cluster_vec_dict = cluster_vec_dict
        self.domain = domain

    def add_attribute(self, atype, value):
        '''Adds an attribute to the node
        atype is the type of attribute, value is the literal value        
        '''
        self.attributes.add((atype, value))
        self.atypes.add(atype)

        try:
            self.domain.dirty = True
        except:
            pass

    def remove_attribute(self, atype, value):
        '''Removes an attribute from the node
        atype is the type of attribute, value is the literal value        
        '''
        self.attributes.remove((atype, value))
        #remove attribute type if last of its kind
        if atype not in {x for x,v in self.attributes}:
            self.atypes.remove(atype)

        try:
            self.domain.dirty = True
        except:
            pass

    def add_predecessor(self, rtype, pred):
        '''Adds a predecessor relationship (incoming connection)'''
        if (rtype, pred) not in self.incoming_relations:
            self.incoming_relations.add((rtype, pred))
            self.rtype_count[rtype] += 1
            self.i_rtypes.add(rtype)

        try:
            self.domain.dirty = True
        except:
            pass

    def remove_predecessor(self, rtype, pred):
        '''Removes a predecessor relationship (incoming connection)'''
        if (rtype, pred) in self.incoming_relations:
            self.incoming_relations.remove((rtype, pred))
            self.rtype_count[rtype] -= 1
            if self.rtype_count[rtype] == 0:
                self.i_rtypes.remove(rtype)
                del self.rtype_count[rtype]

        try:
            self.domain.dirty = True
        except:
            pass

    def add_relation(self, rtype, dest):
        '''Adds a neighbor relationship (outgoing connection)'''
        if (rtype, dest) not in self.outgoing_relations:
            self.outgoing_relations.add((rtype, dest))
            self.rtypes.add(rtype)
            self.rtype_count[rtype] += 1

        try:
            self.domain.dirty = True
        except:
            pass

    def remove_relation(self, rtype, dest):
        '''Removes a neighbor relationship (outgoing connection)'''
        if (rtype, dest) in self.outgoing_relations:
            self.outgoing_relations.remove((rtype, dest))
            self.rtype_count[rtype] -= 1
            if self.rtype_count[rtype] == 0:
                self.rtypes.remove(rtype)
                del self.rtype_count[rtype]

        try:
            self.domain.dirty = True
        except:
            pass

    def __repr__(self):
        return "<%s>(%d)" % (self.name, self.knowledge_level)


class DomainException(Exception):
    pass

class Domain:
    '''Represents a graph about a particular domain
    
    Args:
        nodes - a list of Node objects
        index_metric - which metric function to use for vector indexing
            current options are jaccard_index, dice_coefficient, kulczynski_2
        consolidator - function to consolidate relation types, or None

    '''
    def __init__(self, nodes=None, index_metric=jaccard_index, consolidator=None):
        if nodes != None:
            #mapping between the name of each node and its object
            self.nodes = {n.name:n for n in nodes}
        else:
            self.nodes = {}

        #the function to use for indexing the relationship type vectors
        self.index_metric = index_metric
        self.dirty = False
        if len(self.nodes) > 0:
            #build the graph metadata
            self.rebuild_graph_data(consolidator)

    @property
    def size(self):
        """Return the number of nodes and edges in the graph"""
        return (len(self.nodes), sum([len(x.outgoing_relations) for x in self.nodes.values()]))

    def add_node(self, node):
        """Adds a node object <node> to the map of nodes"""
        self.nodes[node.name] = node
        self.dirty = True

    def remove_node(self, node):
        """Removes a node object <node> from the domain"""
        self.nodes[node.name] = node
        self.dirty = True

    def add_edge(self, rtype, node1, node2):
        """Adds an edge of type <rtype> from <node1> to <node2>"""
        self.nodes[node1].add_relation(rtype,node2)
        self.nodes[node2].add_predecessor(rtype,node1)
        self.dirty = True

    def remove_edge(self, rtype, node1, node2):
        """Removes an edge of type <rtype> from <node1> to <node2>"""
        self.nodes[node1].remove_relation(rtype,node2)
        self.nodes[node2].remove_predecessor(rtype,node1)
        self.dirty = True

    def rebuild_graph_data(self, consolidator=None):
        """rebuild all of the graph data structures

        If consolidator is specified, all relationship types will be 
        consolidated.
        
        <consolidator> is a function which takes a relationship as input
        and returns a relationship as output. Used to cut down and group
        similar relationship types.

        Example usage:
        consolidator("largestCity") = largestcity
        consolidator("derivatives") = derivative
        """

        if len(self.nodes) == 0:
            raise DomainException("No nodes supplied to graph!")

        if consolidator != None:
            for node in self.nodes.values():
                na = set()
                nat = set()
                no = set()
                nr = set()
                ni = set()
                nir = set()
                nrc = Counter()
                for atype, attribute in node.attributes:
                    try:
                        atype = consolidator(atype)
                    except ConsolidatorException:
                        continue
                    na.add((atype, attribute))
                    nat.add(atype)

                for rtype, dest in node.outgoing_relations:
                    try:
                        rtype = consolidator(rtype)
                    except ConsolidatorException:
                        continue
                    no.add((rtype, dest))
                    nr.add(rtype)
                    nrc[rtype] += 1
                    
                for rtype, pred in node.incoming_relations:
                    try:
                        rtype = consolidator(rtype)
                    except ConsolidatorException:
                        continue
                    ni.add((rtype, pred))
                    nir.add(rtype)
                    nrc[rtype] += 1

                #update values
                node.attributes = na
                node.outgoing_relations = no
                node.incoming_relations = ni
                node.rtypes = nr
                node.i_rtypes = nir
                node.atypes = nat
                node.rtype_count = nrc

        # ==== compute member variables ====
        self.usage_map = self.map_uses()
        self.usage_counts = {x:len(y) for x,y in self.usage_map.items()}
        self.rtype_vectors = self.index_rtypes()
        self.node_vectors = self.index_nodes()
        self.rkdtree_keys, _rvalues = zip(*self.rtype_vectors.items())
        self.rkdtree = cKDTree(_rvalues)
        self.nkdtree_keys, _nvalues = zip(*self.node_vectors.items())
        self.nkdtree = cKDTree(_nvalues)

        # ==== precompute some vector constructs ====
        for node in self.nodes.values():
            node.compute_dicts(self)

        # ==== compute tf-idf weights for all nodes ====

        #calculate number of nodes containing rtype and 
        #find maximum frequency rtype for any single node
        maxftd = 0
        c2 = Counter()
        for y in self.nodes.values():
            for k,z in y.rtype_count.items():
                c2[k] += 1
                if z > maxftd:
                    maxftd = z

        #calculate augmented term frequency
        tf = Counter()
        for x,y in self.nodes.items():
            for z,v in y.rtype_count.items():
                tf[(x,z)] = 0.5 + 0.5*(v/maxftd)

        #calculate inverse document frequency
        idf = Counter()
        N = len(self.nodes)
        for x in c2:
            idf[x] = log(N / c2[x])

        tfidf = {}
        for x,y in self.nodes.items():
            for z in y.rtype_count:
                tmp = tfidf.setdefault(x,{})
                tmp[z] = tf[(x,z)] * idf[z]

        self.tfidf = tfidf
        self.dirty = False

    def map_uses(self):
        """Create map between relationship type and all of its uses
        
        Also adds references to node incoming relationships for faster lookup 
        and checks for consistent connections
        """
        out = {}
        for node in self.nodes.values():
            baddies = set()#track incomplete connections and relegate to attributes
            for rtype, dest in node.outgoing_relations:
                try:
                    self.nodes[dest].add_predecessor(rtype, node.name)
                    out.setdefault(rtype, set()).add((node.name, dest))
                except KeyError:
                    baddies.add((rtype, dest))
            for rtype, dest in baddies:
                node.remove_relation(rtype, dest)
                node.add_attribute(rtype, dest)

            atc = node.attributes.copy()
            #check if any attributes have corresponding nodes
            for atype, attrib in atc:
                if attrib in self.nodes:
                    node.remove_attribute(atype, attrib)
                    node.add_relation(atype, attrib)
                    self.nodes[attrib].add_predecessor(atype, node.name)
                    out.setdefault(atype, set()).add((node.name, attrib))
        
        return out   

    def get_closest_relationship(self, point, n=1):
        """
        Returns the closest relationship to a given point as well as the distance

        If n is specified, will return the n closest relationships
        
        """
        n = min(n,len(self.rtype_vectors))#prevent index error
        if n > 1:
            tmp = zip(*self.rkdtree.query(point,n))
            return [(d, self.rkdtree_keys[i]) for d,i in tmp]
        else:
            dist, id = self.rkdtree.query(point,n)
            return [(dist, self.rkdtree_keys[id])]

    def get_closest_node(self, point, n=1):
        """
        Returns the closest node to a given point as well as the distance

        If n is specified, will return the n closest nodes
        
        """
        n = min(n,len(self.nodes))#prevent index error
        if n > 1:
            tmp = zip(*self.nkdtree.query(point,n))
            return [(d, self.nkdtree_keys[i]) for d,i in tmp]
        else:
            dist, id = self.nkdtree.query(point,n)
            return [(dist, self.nkdtree_keys[id])]

    def index_nodes(self):
        """Construct vector representations for every node"""
        out = {}

        #avg = np.mean(list(self.rtype_vectors.values()),axis=0)


        #for name, node in self.nodes.items():
        #    tmp1 = [self.rtype_vectors[rtype]
        #            for rtype, dest in node.outgoing_relations] or [NULL_VEC()]
        #    tmp2 = [permute_rtype_vector(self.rtype_vectors[rtype])
        #            for rtype, prev in node.incoming_relations] or [NULL_VEC()]

        #    net = tmp1 + tmp2

        #    #out[name] = np.asarray(net).mean(axis=0)
        #    #out[name] = np.asarray(net).sum(axis=0)
        #    v = np.asarray(net).sum(axis=0)
        #    if v.any():
        #        out[name] = v/max(v)#softmax(v/max(v))
        #    else:
        #        out[name] = v


        #avg = np.mean(list(out.values()),axis=0)

        #maxm = np.max(list(out.values()),axis=0)

        ####normalize everything
        #for r,v in out.items():
        #    if v.any():
        #        #out[r] = v / sqrt(v.dot(v))
        #        out[r] = softmax((v-avg)/maxm)



        # PCA method 0001701
        rmap = self.rtype_vectors
        data = np.zeros((len(self.nodes), JACCARD_DIMENSIONS), dtype=np.float)
        ix = 0
        for node in self.nodes.values():

            #compute weighted average of each relation type
            tmp = [rmap[rtype] for 
                        rtype, dest in node.outgoing_relations] + \
                  [permute_rtype_vector(rmap[rtype]) for 
                        rtype, prev in node.incoming_relations]

            v = np.asarray(tmp).mean(axis=0) if tmp else NULL_VEC()

            #normalize
            if v.any():
                data[ix] = v / sqrt(v.dot(v))
            else:
                data[ix] = v
            ix += 1

        #eliminate projection onto first 7 principal components
        d2 = data - PCA(data, 7)

        #order of nodes is preserved
        for i,v in enumerate(self.nodes):
            out[v] = softmax(d2[i])

        return out

    def index_rtypes(self):
        """Constructs vector representations for every type of relationship
        in the domain.        
        """
        metric = self.index_metric
        out = {}
        for fnode in self.nodes.values():
            # only consider outgoing relationships because looping over
            # all object anyways, so will cover everything

            for (rtype, dest) in fnode.outgoing_relations:
                dnode = self.nodes[dest]

                # merge outgoing and attributes - distinction should not change
                # how vectors are formed
                a1 = fnode.rtypes | fnode.atypes
                b1 = dnode.rtypes | dnode.atypes
                c1 = a1 - b1
                d1 = b1 - a1
                e1 = b1 & a1
                f1 = b1 ^ a1
                g1 = b1 | a1

                # merge outgoing and attributes - distinction should not change
                # how vectors are formed
                #a2 = {b for a,b in fnode.outgoing_relations} | {b for a,b in fnode.attributes}
                #b2 = {b for a,b in dnode.outgoing_relations} | {b for a,b in dnode.attributes}
                #c2 = a2 - b2
                #d2 = b2 - a2
                #e2 = b2 & a2
                #f2 = b2 ^ a2
                #g2 = b2 | a2

                rval = out.setdefault(rtype, NULL_VEC())

                """
                TODO: add similarity measure between node and prototype nodes

                Idea is to get a ground-truth value for the rtype by measuring
                how src --<rtype>--> dest compares to prototype transformations

                
                
                """

                #types only
                score = np.array([metric(a1, b1),
                                  metric(a1, c1),#1
                                  metric(a1, e1),#2
                                  metric(a1, f1),#3
                                  metric(a1, g1),#4
                                  metric(b1, d1),#1
                                  metric(b1, e1),#2
                                  metric(b1, f1),#3
                                  metric(b1, g1),#4
                                  metric(c1, d1),
                                  metric(c1, f1),#5
                                  metric(d1, f1),#5
                                  metric(f1, g1),
                                  ],dtype=np.float)

           
                #types and objects
                #score = np.array([metric(a1, b1),
                #                  metric(a1, c1),
                #                  metric(a1, e1),
                #                  metric(a1, f1),
                #                  metric(a1, g1),
                #                  metric(b1, d1),
                #                  metric(b1, e1),
                #                  metric(b1, f1),
                #                  metric(b1, g1),
                #                  metric(c1, d1),
                #                  metric(c1, f1),
                #                  metric(c1, c2),
                #                  metric(c1, e2),
                #                  metric(d1, f1),
                #                  metric(d1, d2),
                #                  metric(d1, e2),
                #                  metric(f1, g1),
                #                  metric(a2, b2),
                #                  metric(a2, c2),
                #                  metric(a2, e2),
                #                  metric(a2, f2),
                #                  metric(a2, g2),
                #                  metric(b2, d2),
                #                  metric(b2, e2),
                #                  metric(b2, f2),
                #                  metric(b2, g2),
                #                  metric(c2, f2),
                #                  metric(d2, f2),
                #                  metric(f2, g2)],dtype=np.float)

                out[rtype] = rval + score

        #avg = np.mean(list(out.values()),axis=0)

        #maxm = np.max(list(out.values()),axis=0)


        
        #with open("rrw.pkl","wb+") as f:
        #    pickle.dump(out, f, -1)

        #normalize everything
        for r,v in out.items():
            #out[r] = v / max(v)
            out[r] = v / sqrt(v.dot(v))
            #out[r] = softmax(v/maxm)
            #out[r] = softmax(v/max(v))
            #out[r] = softmax((v-avg)/maxm)

        #for debugging purposes
        #np.save("utils/vectest.npy",np.array(list(out.values())))
        

        '''
        rcount = self.usage_counts
        vs1 = {}
        for rtype, vec in out.items():
            vs1[rtype] = softmax(vec/rcount[rtype])

        data = np.array(list(vs1.values()))
        d2 = data - PCA(data, 1)#eliminate projection onto first principal component

        for i,v in enumerate(vs1):#iteration order is preserved
            #rescale output
            out[v] = softmax(d2[i]/rcount[v])
        '''
        return out

    def serialize(self):
        """Returns a JSON representation of the domain
        
        Format:

        {"idmap":{<node_id>:<node_name>},
         "nodes":[{"name":<node_name>,
                   "text":<node_description>,
                   "neighbors":[["relation",
                                 <relation_type>,
                                 <node_id>],
                                ["literal",
                                <literal_type>,
                                <literal_value>]
                               ]
                   }
                 ]        
        }
        """
        out = {"nodes":[],
               "idmap":{i:x for i,x in enumerate(sorted(self.nodes))}} #map for decoding
        r_idmap = {x:i for i,x in out["idmap"].items()} #map for encoding
        for name, node in self.nodes.items():
            tmp = {"name": name,
                   "text": node.text,
                   "neighbors": []}
            for rtype, dest in node.outgoing_relations:
                if dest in r_idmap:
                    tmp["neighbors"].append(["relation", rtype, str(r_idmap[dest])])
                else:
                    tmp["neighbors"].append(["literal", rtype, dest])
            for atype, attribute in node.attributes:
                tmp["neighbors"].append(["literal", atype, attribute])
            out["nodes"].append(tmp)
        return json.dumps(out,sort_keys=True)

def deserialize(data, consolidator=None):
    """Returns a Domain object constructed from JSON data
    
    Expected Format:

        {"idmap":{<node_id>:<node_name>},
         "nodes":[{"name":<node_name>,
                   "text":<node_description>,
                   "neighbors":[["relation",
                                 <relation_type>,
                                 <node_id>],
                                ["literal",
                                <literal_type>,
                                <literal_value>]
                               ]
                   }
                 ]        
        }
    """
    tmp = json.loads(data)
    nodelist = []
    for n in tmp["nodes"]:
        node = Node(n["name"])
        node.text = n["text"]
        for neighbor in n["neighbors"]:
            if neighbor[0] == "relation":
                node.add_relation(neighbor[1], tmp["idmap"][str(neighbor[2])])
                node
            elif neighbor[0] == "literal":
                node.add_attribute(neighbor[1], str(neighbor[2]))
        nodelist.append(node)
    return Domain(nodelist, consolidator=consolidator)

class DomainLoader:
    """
    Wrapper for loading domains
    
    """
    def __init__(self, filename=None, rawdata=None, cachefile=None, consolidator=None):
        self.nodelist = [] #list of all nodes in the domain
        self.domain_obj = None
        self.consolidator = consolidator
        #load from cache if possible
        if cachefile != None and os.path.isfile(cachefile):
            self.cache_load(cachefile)
        #convenience method to load file in constructor
        #load if not already loaded from cache
        #otherwise store in cache file
        if self.domain_obj == None and (filename or rawdata):
            self.import_data(filename=filename, rawdata=rawdata)
            if cachefile:
                self.cache_store(cachefile)

        

    def import_data(self, filename=None, rawdata=None, append=False):
        """if append is true, will join together multiple data sources
           otherwise will overwrite
        """
        
        if filename:
            with open(filename,"r") as f:
                data = f.read()
        elif rawdata:
            data = rawdata
        else:
            raise Exception("No data given")

        if not append:
            self.nodelist = []

        d = deserialize(data, self.consolidator)
        self.nodelist += list(d.nodes.values())
        if append:
            self.domain_obj = None #mark as outdated
        else:
            self.domain_obj = d

    def cache_store(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(filename, "wb+") as f:
            pickle.dump(self.domain, f, -1)

    def cache_load(self, filename):
        with open(filename, "rb") as f:
            self.domain_obj = pickle.load(f)

    def export_data(self):
        return self.domain.serialize()

    @property
    def domain(self, metric=jaccard_index):
        #if object is marked as outdated, update
        if self.domain_obj == None:
            self.domain_obj = Domain(self.nodelist, metric, self.consolidator)
        return self.domain_obj



class AIMind(DomainLoader):
    """
    Wrapper for AIMind file format
    
    """

    def import_data(self, filename=None, rawdata=None, append=False):
        """if append is true, will join together multiple data sources
           otherwise will overwrite"""
        if filename:
            tree = ET.parse(filename)
        elif rawdata:
            tree = ET.ElementTree(ET.fromstring(rawdata))
        else:
            raise Exception("No data given")

        root = tree.getroot()
        features = root.find("Features")

        if not append:
            self.nodelist = []

        feature_id_table = {}

        # map all feature ids to name
        for feature in features.iter('Feature'):
            feature_id_table[feature.attrib["id"]] = feature.attrib["data"]

        # build relation structure
        for feature in features.iter('Feature'):
            fobj = Node(feature.attrib["data"])
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
                    feature_id_table[neighbor.attrib['dest']])
            self.nodelist.append(fobj)

    def export_data(self):
        #TODO: implement this
        pass