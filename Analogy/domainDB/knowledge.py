"""
knowledge.py

Contains functions for managing the domain files.
"""

from os.path import join, isfile, abspath, exists
from os import listdir, makedirs
import hashlib
import base64
import json
import asyncio
import random

from .database import init_db
from .models import Concept, Domain, Unknown
from ..utils.DBpediaCrawler import keyword_search, generate_graph, get_label, make_uri
from ..utils.utils import deserialize

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
        if not exists(self.datapath):
            makedirs(self.datapath)

    def get_random_concept(self):
        try:
            session = self.database()
            query = session.query(Concept)
            rowCount = int(query.count())
            tmp = query.offset(int(rowCount*random.random())).first()
            return tmp.name, session.query(Domain.filepath).filter(Domain.id==tmp.domain).first().filepath
        finally:
            self.database.remove()

    def get_random_domain(self):
        try:
            session = self.database()
            query = session.query(Domain)
            rowCount = int(query.count())
            return query.offset(int(rowCount*random.random())).first().filepath
        finally:
            self.database.remove()

    def get_uri(self, concept):
        return keyword_search(concept)

    def find_domains(self, concept, explicit=True, ordered=False, uri=False):
        """Return the valid keyword used and the domains containing that keyword, 
        an Unkown object if it is not yet known, or None if it is not in DBpedia
        
        if explicit is True, the DBpedia query must be an exact match

        if ordered is True, it will sort the domains by their size.

        if uri is True, concept is assumed to be a DBpedia URI already
        
        """

        session = self.database()
        def find_helper():
            if not uri:
                ret = keyword_search(concept)
                #check for exact match
                if explicit and get_label(ret) != concept:
                    return get_label(ret), None
            else:
                ret = make_uri(concept)
            if ret:
                domains = session.query(Concept.domain, Domain.filepath, Domain.details).join(Domain).filter(Concept.name == get_label(ret))
                if domains.count() == 0:
                    #if the topic is not yet known, add to list of unknown topics
                    ukn = Unknown.query.filter_by(name=ret).first()
                    if ukn == None:
                        ukn = Unknown()
                        ukn.name = ret
                        session.add(ukn)
                        session.commit()
                    return get_label(ret), ukn  
                else:

                    tmpd = [x for x in domains.all()]
                    if ordered:
                        tmpd.sort(key=lambda x: int(json.loads(x.details).get("size") or 100), reverse=True)
                    return get_label(ret), [x.filepath for x in tmpd]
            else:
                #if the topic is not in DBpedia, return None
                return None, None
        tmp = find_helper()
        self.database.remove()
        return tmp

    def refresh_database(self, domain=None):
        """Check the data file folder for domain files and update the database
        If domain is None, it will check all files in folder.
        Domain must be an absolute path.
        """

        session = self.database()

        if domain != None:
            domains = [domain]
        else:
            domains = [join(self.datapath, f) for f in listdir(self.datapath) if isfile(join(self.datapath, f))]
            
        for fname in domains:
            d = Domain.query.filter_by(filepath=fname).first()
            if d != None:
                #clear all old concepts
                Concept.query.filter_by(domain=d.id).delete()
                with open(fname, "r") as f:
                    data = deserialize(f.read())
                    for concept in data.nodes:
                        c = Concept()
                        c.domain = d.id
                        c.name = concept
                        session.add(c)
                    session.commit()
            else:
                print("no db for %s"%fname)
        print("Database refreshed.")
        self.database.remove()



    def reconcile_knowledge(self, limit=100):
        """For each unknown topic, check if it is now known. If not, search for it."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        session = self.database()
        unknowns = session.query(Unknown)
        total = unknowns.count()
        filepaths = []
        for i,u in enumerate(unknowns.all()):
            print("reconciling unknown %d/%d: "%(i+1,total), u.name)
            if u.name[:19] != "http://dbpedia.org/": #assume proper dbpedia URI
                ret = keyword_search(u.name)
            else:
                ret = u.name
            if ret != None:
                tmpd = self.generate_domain(ret, limit)
                if tmpd != None: 
                    session.delete(u)
                    session.commit()
                    filepaths.append(tmpd.filepath)
            else:
                print("Error: could not find DBpedia entry for %s"%u.name)

        self.database.remove()
        loop.stop()
        return filepaths

    def consolidate_domain(self, domain):
        """Re-cluster domain file, if necessary"""
        raise NotImplementedError()

    def consolidate_domains(self):
        """Re-cluster all domain files"""
        raise NotImplementedError()

    def generate_domain(self, uri, num_nodes=100, _re=False):
        """Generate a domain centered on a concept. Expects a DBpedia URI."""
        session = self.database()
        def helper_func():
            nonlocal uri
            fname = join(self.datapath, shorten(uri))
            d = Domain.query.filter_by(filepath=fname).first()
            if isfile(fname):
                if d == None:
                    print("Domain %s exists but is not in database. Adding."%fname)
                    d = Domain()
                    d.filepath = fname
                    d.details = json.dumps({"root_uri":uri,"size":num_nodes})
                    session.add(d)
                    session.commit()
                else:
                    print("Domain %s already exists."%fname)
                return d
            else:
                try:
                    G = generate_graph(uri, num_nodes)
                    #if disambiguation page on exact uri, try search
                    if len(G.nodes) == 0:
                        if not _re:
                            return self.generate_domain(keyword_search(get_label(uri)), num_nodes, True)
                        else:
                            print("Error: could not generate domain for concept: %s"%(uri))
                            return None
                except Exception as e:
                    print("Error generating domain for concept: %s > %s"%(uri,e))
                    return None
                with open(fname,"w+") as f:
                    print("Domain generated for concept: %s"%uri)
                    f.write(G.serialize())
                d = Domain()
                d.filepath = fname
                d.details = json.dumps({"root_uri":uri,"size":num_nodes})
                session.add(d)
                session.commit()
                self.refresh_database(fname)
                return d
        tmp = helper_func()
        self.database.remove()
        return tmp

    def list_unknowns(self):
        return self.database.query(Unknown).all()
