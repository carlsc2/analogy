"""
knowledge.py

Contains functions for managing the domain files.
"""

from domainDB.database import init_db
from domainDB.models import Concept, Domain, Unknown
from utils.DBpediaCrawler import keyword_search, generate_graph, get_label
from utils.utils import deserialize
from os.path import join, isfile, abspath
from os import listdir
import hashlib
import base64
import json

KEY_SIZE = 32 #the size of the file names to generate 

def shorten(url):
    #shorten a url
    orig_id = url.encode()
    shorter_id = base64.urlsafe_b64encode(hashlib.md5(orig_id).digest())[:KEY_SIZE]
    if type(shorter_id) == bytes:
        shorter_id = shorter_id.decode()
    return shorter_id

class DomainManager:
    def __init__(self, db_file, datapath):
        """
        Creates an instance to manage domain files

        Parameters:
            db_file <string>: path to manager database
            datapath <string>: directory where domain files are stored
        """
        self.database = init_db(db_file)
        self.datapath = abspath(datapath)

    def find_domains(self, concept, explicit=True):
        """Return the domains containing a topic, an Unkown object if it is not yet known,
        or None if it is not in DBpedia
        
        if explicit is True, the DBpedia query must be an exact match
        
        """
        ret = keyword_search(concept)
        if ret:
            name = get_label(ret)
            #check for exact match
            if explicit and name != concept:
                return None
            domains = self.database.query(Concept.domain, Domain.filepath).join(Domain).filter(Concept.name == name)
            if domains.count() == 0:
                #if the topic is not yet known, add to list of unknown topics
                ukn = Unknown.query.filter_by(name=name).first()
                if ukn == None:
                    ukn = Unknown()
                    ukn.name = name
                    self.database.add(ukn)
                    self.database.commit()
                return ukn  
            else:
                return [x.filepath for x in domains.all()]
        else:
            #if the topic is not in DBpedia, return None
            return None  

    def refresh_database(self, domain=None):
        """Check the data file folder for domain files and update the database
        If domain is None, it will check all files in folder.
        Domain must be an absolute path.
        """

        if domain != None:
            domains = [domain]
        else:
            domains = [join(self.datapath, f) for f in listdir(self.datapath) if isfile(join(self.datapath, f))]
            
        for fname in domains:
            d = Domain.query.filter_by(filepath=fname).first()
            if d != None:
                print(d)
                #clear all old concepts
                Concept.query.filter_by(domain=d.id).delete()
                with open(fname, "r") as f:
                    data = deserialize(f.read())
                    for concept in data.nodes:
                        print(concept)
                        c = Concept()
                        c.domain = d.id
                        c.name = concept
                        self.database.add(c)
                    self.database.commit()
            else:
                print("no db for %s"%fname)
        print("Database refreshed.")



    def reconcile_knowledge(self):
        """For each unknown topic, check if it is now known. If not, search for it."""
        unknowns = self.database.query(Unknown)
        total = unknowns.count()
        for i,u in enumerate(unknowns.all()):
            print("reconciling unknown %d/%d: "%(i+1,total), u.name)
            ret = keyword_search(u.name)
            if ret != None:
                if self.generate_domain(ret) != None: 
                    self.database.delete(u)
                    self.database.commit()
            else:
                print("Error: could not find DBpedia entry for %s"%u.name)

    def consolidate_domain(self, domain):
        """Re-cluster domain file, if necessary"""
        raise NotImplementedError()

    def consolidate_domains(self):
        """Re-cluster all domain files"""
        raise NotImplementedError()

    def generate_domain(self, uri, num_nodes=100):
        """Generate a domain centered on a concept. Expects a DBpedia URI."""
        fname = join(self.datapath, shorten(uri))
        d = Domain.query.filter_by(filepath=fname).first()
        if isfile(fname):
            if d == None:
                print("Domain %s exists but is not in database. Adding."%fname)
                d = Domain()
                d.filepath = fname
                d.details = json.dumps({"root_uri":uri})
                self.database.add(d)
                self.database.commit()
            else:
                print("Domain %s already exists."%fname)
            return d
        else:
            try:
                G = generate_graph(uri, num_nodes)
            except:
                print("Error generating domain for concept: %s"%uri)
                return None
            with open(fname,"w+") as f:
                print("Domain generated for concept: %s"%uri)
                f.write(G.serialize())
            d = Domain()
            d.filepath = fname
            d.details = json.dumps({"root_uri":uri})
            self.database.add(d)
            self.database.commit()
            self.refresh_database(fname)
            return d

    def list_unknowns(self):
        return self.database.query(Unknown).all()