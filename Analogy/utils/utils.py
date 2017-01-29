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

def permute_rtype_vector(x):
    """convert incoming relationship to outgoing and vice versa"""
    return np.array([x[0],x[5],x[6],x[7],x[8],x[1],x[2],x[3],x[4],x[9],
                     x[13],x[14],x[15],x[10],x[11],x[12],x[16],x[17]],dtype=np.float)

JACCARD_DIMENSIONS = 18
NULL_VEC = lambda : np.zeros(JACCARD_DIMENSIONS)
NULL_VEC2 = lambda : np.zeros(JACCARD_DIMENSIONS * 2)



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
        self.outgoing_relations = set()  # set of relations to other nodes
        self.incoming_relations = set()  # set of relations from other nodes
        self.rtypes = set() #set of types of outgoing relationships
        self.rtype_count = Counter() #how many times each rtype is used
        self.knowledge_level = len(self.outgoing_relations) +\
                               len(self.incoming_relations)
        self.text = ""

    def get_rtype_ratios(self):
        total = sum(self.rtype_count.values())
        return {x:self.rtype_count[x]/total for x in self.rtype_count}

    def add_predecessor(self, rtype, pred):
        '''Adds a predecessor relationship (incoming connection)
        
        Note: This should not be called directly if the feature is already in
        a Domain
        '''
        self.incoming_relations.add((rtype, pred))
        self.rtype_count[rtype] += 1
        self.knowledge_level = len(self.outgoing_relations) +\
                               len(self.incoming_relations)

    def add_relation(self, rtype, dest):
        '''Adds a neighbor relationship (outgoing connection)
        
        Note: This should not be called directly if the feature is already in
        a Domain
        '''
        self.outgoing_relations.add((rtype, dest))
        self.rtypes.add(rtype)
        self.rtype_count[rtype] += 1
        self.knowledge_level = len(self.outgoing_relations) +\
                               len(self.incoming_relations)

    def __repr__(self):
        return "<%s>(%d)" % (self.name, self.knowledge_level)

class Domain:
    '''Represents a graph about a particular domain
    
    Args:
        nodes - a list of Node objects
        index_metric - which metric function to use for vector indexing
            current options are jaccard_index, dice_coefficient, kulczynski_2

    '''
    def __init__(self, nodes=None, index_metric=jaccard_index):
        if nodes != None:
            #mapping between the name of each node and its object
            self.nodes = {n.name:n for n in nodes}
        else:
            self.nodes = {}
        #the function to use for indexing the relationship type vectors
        self.index_metric = index_metric
        #maps each relationship type to all its uses
        self.usage_map = self.map_uses()
        #maps each rtype to its vector
        self.rtype_vectors = self.index_rtypes()
        #maps each node to its vector
        self.node_vectors = self.index_nodes()
        if len(self.nodes) > 0:
            #nearest rtype fast lookup
            self.rkdtree_keys, _rvalues = zip(*self.rtype_vectors.items())
            self.rkdtree = cKDTree(_rvalues)
            #nearest node fast lookup
            self.nkdtree_keys, _nvalues = zip(*self.node_vectors.items())
            self.nkdtree = cKDTree(_nvalues)
        #dirty flag -- if graph has changed, should re-evaluate things
        self.dirty = False

    @property
    def size(self):
        """Return the number of nodes and edges in the graph"""
        return (len(self.nodes), sum([len(x.outgoing_relations) for x in self.nodes.values()]))

    def add_node(self, node):
        """Adds a node object <node> to the map of nodes"""
        self.nodes[node.name] = node
        self.dirty = True

    def add_edge(self, rtype, node1, node2):
        """Adds an edge of type <rtype> from <node1> to <node2>"""
        self.nodes[node1].add_relation(rtype,node2)
        self.nodes[node2].add_predecessor(rtype,node1)
        self.dirty = True

    def rebuild_graph_data(self):
        """rebuild all of the graph data structures"""
        self.usage_map = self.map_uses()
        self.rtype_vectors = self.index_rtypes()
        self.node_vectors = self.index_nodes()
        self.rkdtree_keys, _rvalues = zip(*self.rtype_vectors.items())
        self.rkdtree = cKDTree(_rvalues)
        self.nkdtree_keys, _nvalues = zip(*self.node_vectors.items())
        self.nkdtree = cKDTree(_nvalues)
        self.dirty = False

    def index_nodes(self):
        """Construct vector representations for every node"""
        out = {}
        for name, node in self.nodes.items():
            tmp1 = [self.rtype_vectors[rtype]
                    for rtype, dest in node.outgoing_relations] or [NULL_VEC()]
            tmp2 = [permute_rtype_vector(self.rtype_vectors[rtype])
                    for rtype, prev in node.incoming_relations] or [NULL_VEC()]

            net = tmp1 + tmp2

            out[name] = np.asarray(net).mean(axis=0)

        ##normalize everything
        #for r,v in out.items():
        #    out[r] = v / sqrt(v.dot(v))

        return out

    def map_uses(self):
        """Create map between relationship type and all of its uses
        
        Also adds references to node incoming relationships for faster lookup 
        """
        out = {}
        for node in self.nodes.values():
            for rtype, dest in node.outgoing_relations:
                out.setdefault(rtype, set()).add((node.name, dest)) 
                self.nodes[dest].add_predecessor(rtype, node.name)  
        return out   

    def get_closest_relationship(self, point, n=1):
        """
        Returns the closest relationship to a given point as well as the distance

        If n is specified, will return the n closest relationships
        
        """
        if n > 1:
            tmp = zip(*self.rkdtree.query(point,n))
            return [(d, self.rkdtree_keys[i]) for d,i in tmp]
        else:
            dist, id = self.rkdtree.query(point,n)
            return [dist, self.rkdtree_keys[id]]

    def get_closest_node(self, point, n=1):
        """
        Returns the closest node to a given point as well as the distance

        If n is specified, will return the n closest nodes
        
        """
        if n > 1:
            tmp = zip(*self.nkdtree.query(point,n))
            return [(d, self.nkdtree_keys[i]) for d,i in tmp]
        else:
            dist, id = self.nkdtree.query(point,n)
            return [dist, self.nkdtree_keys[id]]

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
                a = fnode.rtypes
                b = dnode.rtypes
                c = a - b
                d = b - a
                e = b & a
                f = b ^ a
                g = b | a
                h = c | d

                """
                TODO: add similarity measure between node and prototype nodes

                Idea is to get a ground-truth value for the rtype by measuring
                how src --<rtype>--> dest compares to prototype transformations

                
                
                """

                rval = out.setdefault(rtype, NULL_VEC())

                score = np.array([metric(a, b),
                                  metric(a, c),
                                  metric(a, e),
                                  metric(a, f),
                                  metric(a, g),
                                  metric(b, d),
                                  metric(b, e),
                                  metric(b, f),
                                  metric(b, g),
                                  metric(c, d),
                                  metric(c, e),
                                  metric(c, f),
                                  metric(c, g),
                                  metric(d, e),
                                  metric(d, f),
                                  metric(d, g),
                                  metric(f, g),
                                  metric(f, h)], dtype=np.float)

                out[rtype] = rval + score

        #normalize everything
        #for r,v in out.items():
        #    out[r] = v / sqrt(v.dot(v))
        np.save("vectest.npy",np.array(list(out.values())))
        return out


class AIMind:
    """
    Wrapper for AIMind file format
    
    """
    def __init__(self,filename=None,rawdata=None):
        self.feature_id_table = {}
        self.featurelist = []
        if filename or rawdata:
            self.import_data(filename=filename,rawdata=rawdata)

    def as_domain(self, metric=jaccard_index):
        return Domain(self.featurelist, metric)
        
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
            self.feature_id_table = {}

        # map all feature ids to name
        for feature in features.iter('Feature'):
            self.feature_id_table[feature.attrib["id"]] = feature.attrib["data"]

        self.featurelist = []

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
                    self.feature_id_table[neighbor.attrib['dest']])
            self.featurelist.append(fobj)

    def export_data(self, feature_id_table, domain):
        #TODO: implement this
        pass