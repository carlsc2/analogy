from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import json
import asyncio
import random
import signal, sys
from .utils import Domain, Node
from concurrent.futures import ThreadPoolExecutor
import time
from math import exp
import difflib

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

NUM_WORKERS = 15 #number of worker threads

def generate_graph(seeds, total, depth_limit=None,
                   max_outgoing=None, max_incoming=None,
                   relevance_threshold=None, debug=False):
    '''
    Generates a knowledge graph from seed keywords, up to <total> nodes

    if <depth_limit> is specified, only nodes up to <depth_limit> from the
    seed will be included

    if <max_outgoing> is specified, only <max_outgoing> outgoing connections
    will be searched for each node

    if <max_incoming> is specified, only <max_incoming> incoming connections
    will be searched for each node

    if <relevance_threshold> is specified, links which are proportionally unrelated
    to visited nodes as defined by the threshold will not be explored. Nodes which
    have a relevance score below the threshold are ignored.
    
    '''

    WORKER_COUNT = min(total, NUM_WORKERS)
    
    start_time = time.time()

    if type(seeds) != list:
        seeds = [seeds]
    workers = set()
    visited = set()

    if relevance_threshold != None:
        q = asyncio.PriorityQueue() #queue of URIs to process
    else:
        q = asyncio.Queue()

    print("Generating knowledge graph with seeds:")
    for seed in seeds:
        if seed[:19] == "http://dbpedia.org/": #assume proper dbpedia URI
            uri = seed
        else:
            uri = keyword_search(seed) #search for URI from keyword
            if uri is None:
                print("ERROR: keyword %s not found"%seed)
                return
        print("Seed: ",uri)
        if relevance_threshold != None:
            q.put_nowait((10,(0,uri)))
        else:
            q.put_nowait((0,uri))
        visited.add(uri)#need to add initial to visited
 
    count = 0
    graph = Domain()

    fillcount = 0

    def get_relevance(z):
        #check relevance of combined incoming and outgoing links
        return len(z&visited)/min(len(z), len(visited))

    async def consume(loop, executor):
        nonlocal count, fillcount

        #check for deadlocks
        if q.empty() and fillcount == 0:
            return

        #process next link from queue
        if relevance_threshold != None:
            priority, (depth, value) = await q.get() #get next URI
        else:
            (depth, value) = await q.get()

        fillcount += 1

        #fetch URI data
        data = await loop.run_in_executor(executor, get_data, value)

        #fetch links
        linkdata = await loop.run_in_executor(executor, get_links, value)

        z = linkdata['incoming'] | linkdata['outgoing']

        #if no actual links, don't add to graph
        if len(z) == 0:
            fillcount -= 1
            return

        #check relevance to other nodes
        if relevance_threshold != None:
            r = get_relevance(z)

        

        #add node if relevant enough
        #slowly build up to specified threshold based on depth

        n = Node(get_label(value)) #add node to graph
        for rtype, dest in data.items():
            if dest[:19] == "http://dbpedia.org/":#assume uri is node
                n.add_relation(get_label(rtype), get_label(dest))
            else:
                n.add_attribute(get_label(rtype), dest)
        graph.add_node(n)
        count += 1

        if debug:
            try:
                #check what percent of links tie back
                if relevance_threshold != None:
                    print(value, depth, r)
                else:
                    print(value, depth)
            except UnicodeDecodeError:
                pass

        #stop exploring if too deep
        if depth_limit != None and depth+1 > depth_limit:
            fillcount -= 1
            return

        #don't explore links if node isn't relevant enough
        if relevance_threshold != None and (depth > 0 and r < relevance_threshold):
            if debug:
                print("==> Skipping irrelevant node:", depth, value, r)
            fillcount -= 1
            return            

        if max_outgoing != None:
            for link in random.sample(linkdata['outgoing'],
                                      min(max_outgoing,
                                          len(linkdata['outgoing']))):
                if link not in visited:
                    #negative priority so higher relevance first
                    #prioritize outgoing over incoming
                    if relevance_threshold != None:
                        q.put_nowait((-r*1.5/exp(depth),(depth+1,link)))
                    else:
                        q.put_nowait((depth+1,link))
                    visited.add(link)
        else:
            for link in linkdata['outgoing']:
                if link not in visited:
                    #negative priority so higher relevance first
                    #prioritize outgoing over incoming
                    if relevance_threshold != None:
                        q.put_nowait((-r*1.5/exp(depth),(depth+1,link)))
                    else:
                        q.put_nowait((depth+1,link))
                    visited.add(link)

        if max_incoming != None:     
            for link in random.sample(linkdata['incoming'],
                                      min(max_incoming,
                                          len(linkdata['incoming']))):
                if link not in visited:
                    #negative priority so higher relevance first
                    if relevance_threshold != None:
                        q.put_nowait((-r/exp(depth),(depth+1,link)))
                    else:
                        q.put_nowait((depth+1,link))
                    visited.add(link)
        else:
            for link in linkdata['incoming']:
                if link not in visited:
                    #negative priority so higher relevance first
                    if relevance_threshold != None:
                        q.put_nowait((-r/exp(depth),(depth+1,link)))
                    else:
                        q.put_nowait((depth+1,link))
                    visited.add(link)

        fillcount -= 1

        return

    loop = asyncio.get_event_loop()

    # Stop the loop concurrently                                                
    async def exit_loop(loop):                                                                                    
        loop.stop()                                          

    async def grow(loop, executor):
        #grow the graph
        for i in range(WORKER_COUNT):
            workers.add(asyncio.ensure_future(consume(loop,
                                                      executor)))
        while count < total:
            if q.empty() and fillcount == 0:#prevent dead ends
                for worker in workers:
                    worker.cancel()
                asyncio.ensure_future(exit_loop(loop))  
                return
            done,_ = await asyncio.wait(workers,
                                        return_when=asyncio.FIRST_COMPLETED)
            for ret in done:
                workers.remove(ret)
                while len(workers) < WORKER_COUNT:
                    if q.empty() and fillcount == 0:#prevent race condition
                        break
                    if count + len(workers) < total:
                        workers.add(asyncio.ensure_future(consume(loop,
                                                                  executor)))
                    else:
                        break
            if debug:
                try:
                    print(count)
                except UnicodeDecodeError:
                    pass

    try:
        def signal_handler(signal, frame):  
            loop.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except:
        pass

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        try:
            loop.run_until_complete(grow(loop, executor))
        except Exception as e:
            print("Error: ", e)

    print("Graph constructed in %.5f seconds"%(time.time() - start_time))

    try:
        graph.rebuild_graph_data()
    except Exception:
        pass

    return graph

def get_label(uri):
    if uri == None:
        return None
    if uri[:19] == "http://dbpedia.org/":
        #what kind of uri scheme allows slashes??
        return "/".join((uri.split("/")[4:])).replace("_"," ")
    else:
        return uri

def make_uri(concept):
    if concept == None:
        return None
    if concept[:19] == "http://dbpedia.org/":
        return concept
    else:
        return "http://dbpedia.org/resource/" + concept.replace(" ","_")

def keyword_search(keyword, limit=None, similar=False):
    """Queries DBpedia concepts based on a keyword

    Will return the top result based on DBpedia's ranking unless 
    limit is specified, in which case it will return a list of
    <limit> results. Will return None if no results are found.

    If similar is True, it will weigh the quality of the
    results by their word similarity to the keyword.

    Note: By default, similar will look at the top 10 results,
    even if it only returns the best one

    """
    data=urlencode({'QueryString':keyword,
                    'MaxHits':(limit or (10 if similar else 1))})
    req = Request("http://lookup.dbpedia.org/api/search/KeywordSearch?"+data,
                  headers = {'Accept': 'application/json'})
    response = json.loads(urlopen(req,timeout=5).read().decode("utf8"))['results']

    results = [(x['refCount'],x['label'],x['uri'],) for x in response]

    if similar:
        results = sorted([(c/exp(1-difflib.SequenceMatcher(None, keyword.lower(), a.lower()).ratio()),a,b)
                          for c,a,b in results],reverse=True)
        
    if len(results) > 0:
        if limit == None:
            #take first result
            return results[0][2]
        else:
            #return all results if specified
            return [b for c,a,b in results]
    return None

def get_data(uri):
    #gets all data for a given uri
    query = """
        SELECT DISTINCT (?r as ?relationship) (str(?p) as ?property) WHERE {
            <%s> ?r ?p.
            filter not exists {
                <%s> dbo:wikiPageRedirects|dbo:wikiPageDisambiguates ?p
            }
            FILTER regex(?r,'dbpedia.org','i').
            FILTER(!isLiteral(?p) || lang(?p) = '' || langMatches(lang(?p), 'en'))
        }"""%(uri, uri)

    sparql.setQuery(query)
    results = sparql.query().convert()
    ret = {}
    for obj in results['results']['bindings']:
        ret[obj['relationship']['value']] = obj['property']['value']
    return ret

def get_links(uri):
    #gets links to other dbpedia entries for a given uri
    query = """
        SELECT DISTINCT ?r1 ?p1 ?r2 ?p2 WHERE {{
            <%s> ?r1 ?p1.
            filter not exists {
                <%s> dbo:wikiPageRedirects|dbo:wikiPageDisambiguates ?p1
            }
            ?p1 rdfs:label ?pl.
            FILTER regex(?r1,'dbpedia.org','i').
        }UNION{
            ?p2 ?r2 <%s>.
            filter not exists {
                ?p2 dbo:wikiPageRedirects|dbo:wikiPageDisambiguates <%s>
            }
            ?p2 rdfs:label ?pl.
            FILTER regex(?r2,'dbpedia.org','i').
        }}"""%(uri, uri, uri, uri)

    sparql.setQuery(query)
    results = sparql.query().convert()
    ret = {'incoming':set(),
           'outgoing':set()}
    for obj in results['results']['bindings']:
        if 'p1' in obj:
            ret['outgoing'].add(obj['p1']['value'])
        elif 'p2' in obj:
            ret['incoming'].add(obj['p2']['value'])
    return ret
