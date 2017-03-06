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
                     #rtype permutations
    return np.array([x[0],x[5],x[6],x[7],x[8],x[1],x[2],x[3],x[4],x[9],
                     x[13],x[14],x[15],x[10],x[11],x[12],x[16],x[17],
                     #atype permutations
                     x[18],x[23],x[24],x[25],x[26],x[19],x[20],x[21],x[22],x[27],
                     x[31],x[32],x[33],x[28],x[29],x[30],x[34],x[35],
                     ],dtype=np.float)

JACCARD_DIMENSIONS = 36
#JACCARD_DIMENSIONS = 22
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
        self.attributes = set() #set of (attribute, literal value) pairs
        self.outgoing_relations = set()  # set of relations to other nodes
        self.incoming_relations = set()  # set of relations from other nodes
        self.rtypes = set() #set of types of outgoing relationships
        self.atypes = set() #set of types of attributes
        self.rtype_count = Counter() #how many times each rtype is used
        self.knowledge_level = len(self.outgoing_relations) +\
                               len(self.incoming_relations) +\
                               len(self.attributes)
        self.text = ""

    def get_rtype_ratios(self):
        total = sum(self.rtype_count.values())
        return {x:self.rtype_count[x]/total for x in self.rtype_count}

    def add_attribute(self, atype, value):
        '''Adds an attribute to the node
        atype is the type of attribute, value is the literal value        
        '''
        self.attributes.add((atype, value))
        self.atypes.add(atype)

    def remove_attribute(self, atype, value):
        '''Removes an attribute from the node
        atype is the type of attribute, value is the literal value        
        '''
        self.attributes.remove((atype, value))
        #remove attribute type if last of its kind
        if atype not in {x for x,v in self.attributes}:
            self.atypes.remove(atype)

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
                               len(self.incoming_relations) +\
                               len(self.attributes)

    def remove_relation(self, rtype, dest):
        '''Removes a neighbor relationship (outgoing connection)
        
        Note: This should not be called directly if the feature is already in
        a Domain
        '''
        if (rtype, dest) in self.outgoing_relations:
            self.outgoing_relations.remove((rtype, dest))
            self.rtype_count[rtype] -= 1
            if self.rtype_count[rtype] == 0:
                self.rtypes.remove(rtype)
                del self.rtype_count[rtype]
            self.knowledge_level = len(self.outgoing_relations) +\
                                   len(self.incoming_relations) +\
                                   len(self.attributes)

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
                a1 = fnode.rtypes
                b1 = dnode.rtypes
                c1 = a1 - b1
                d1 = b1 - a1
                e1 = b1 & a1
                f1 = b1 ^ a1
                g1 = b1 | a1
                h1 = c1 | d1

                #for all outgoing connections, check difference between attributes
                a2 = fnode.atypes
                b2 = dnode.atypes
                c2 = a2 - b2
                d2 = b2 - a2
                e2 = b2 & a2
                f2 = b2 ^ a2
                g2 = b2 | a2
                h2 = c2 | d2

                """
                TODO: add similarity measure between node and prototype nodes

                Idea is to get a ground-truth value for the rtype by measuring
                how src --<rtype>--> dest compares to prototype transformations

                
                
                """

                rval = out.setdefault(rtype, NULL_VEC())

                score = np.array([metric(a1, b1),
                                  metric(a1, c1),
                                  metric(a1, e1),
                                  metric(a1, f1),
                                  metric(a1, g1),
                                  metric(b1, d1),
                                  metric(b1, e1),
                                  metric(b1, f1),
                                  metric(b1, g1),
                                  metric(c1, d1),
                                  metric(c1, e1),
                                  metric(c1, f1),
                                  metric(c1, g1),
                                  metric(d1, e1),
                                  metric(d1, f1),
                                  metric(d1, g1),
                                  metric(f1, g1),
                                  metric(f1, h1),

                                  metric(a2, b2),
                                  metric(a2, c2),
                                  metric(a2, e2),
                                  metric(a2, f2),
                                  metric(a2, g2),
                                  metric(b2, d2),
                                  metric(b2, e2),
                                  metric(b2, f2),
                                  metric(b2, g2),
                                  metric(c2, d2),
                                  metric(c2, e2),
                                  metric(c2, f2),
                                  metric(c2, g2),
                                  metric(d2, e2),
                                  metric(d2, f2),
                                  metric(d2, g2),
                                  metric(f2, g2),
                                  metric(f2, h2),
                                  ], dtype=np.float)

                out[rtype] = rval + score

        #normalize everything
        for r,v in out.items():
            out[r] = v / sqrt(v.dot(v))
        np.save("utils/vectest.npy",np.array(list(out.values())))
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

def deserialize(data):
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
    d = Domain(nodelist)
    d.rebuild_graph_data()
    return d

class DomainLoader:
    """
    Wrapper for loading domains
    
    """
    def __init__(self,filename=None,rawdata=None):
        self.nodelist = [] #list of all nodes in the domain
        if filename or rawdata:#convenience method to load file in constructor
            self.import_data(filename=filename, rawdata=rawdata)

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

        d = deserialize(data)
        self.nodelist += list(d.nodes.values())

    def export_data(self):
        return self.domain.serialize()

    @property
    def domain(self, metric=jaccard_index):
        return Domain(self.nodelist, metric)



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