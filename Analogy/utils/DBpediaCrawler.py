from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import json
import asyncio
import random
import signal, sys
from utils import Domain, Node
from concurrent.futures import ThreadPoolExecutor
import time

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

NUM_WORKERS = 15
MAX_OUTGOING_LINKS = 100 #maximum number of incoming links to explore per node
MAX_INCOMING_LINKS = 100 #maximum number of incoming links to explore per node

def generate_graph(seed, total):
	'''
	Generates a knowledge graph from a seed keyword, up to <total> nodes
	
	'''

	WORKER_COUNT = min(total, NUM_WORKERS)

	start_time = time.time()
	if seed[:19] == "http://dbpedia.org/": #assume proper dbpedia URI
		uri = seed
	else:
		uri = keyword_search(seed) #search for URI from keyword
		if uri is None:
			print("ERROR: keyword %s not found"%seed)
			return

	workers = set()
	visited = set()
	q = asyncio.Queue() #queue of URIs to process
	q.put_nowait(uri)
	visited.add(uri)
	count = 0
	graph = Domain()

	async def consume(loop, executor):
		nonlocal count
		#process next link from queue
		value = await q.get() #get next URI
		print(value)
		data = await loop.run_in_executor(executor, get_data, value)#fetch URI data

		n = Node(get_label(value)) #add node to graph
		for rtype, dest in data.items():
			n.add_relation(get_label(rtype), get_label(dest))
		graph.add_node(n)
		count += 1

		#explore links
		tmp = await loop.run_in_executor(executor, get_links, value)#fetch URI data

		for link in random.sample(tmp['outgoing'], min(MAX_OUTGOING_LINKS, len(tmp['outgoing']))):
			if link not in visited:
				q.put_nowait(link)
				visited.add(link)
				
		for link in random.sample(tmp['incoming'], min(MAX_INCOMING_LINKS, len(tmp['incoming']))):
			if link not in visited:
				q.put_nowait(link)
				visited.add(link)

		return True

	loop = asyncio.get_event_loop()

	async def grow(loop, executor):
		#grow the graph
		for i in range(WORKER_COUNT):
			workers.add(asyncio.ensure_future(consume(loop, executor)))
		while count < total:
			done,_ = await asyncio.wait(workers,return_when=asyncio.FIRST_COMPLETED)
			for ret in done:
				workers.remove(ret)
				while len(workers) < WORKER_COUNT:
					if count + len(workers) < total:
						workers.add(asyncio.ensure_future(consume(loop, executor)))
					else:
						break
			print(count)

	def signal_handler(signal, frame):  
		loop.stop()
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
		loop.run_until_complete(grow(loop, executor))

	print("Graph constructed in %.5f seconds"%(time.time() - start_time))

	return graph

def get_label(uri):
	if uri[:19] == "http://dbpedia.org/":
		return (uri.split("/")[-1]).replace("_"," ")
	else:
		return uri

def keyword_search(keyword,limit=None):
    data=urlencode({'QueryString':keyword,
                 'MaxHits':(limit or 1)})
    req = Request("http://lookup.dbpedia.org/api/search/KeywordSearch?"+data,
               headers = {'Accept': 'application/json'})
    results = json.loads(urlopen(req,timeout=5).read().decode("utf8"))['results']
    if limit != None:#return all results if specified
        return [x['uri'] for x in results]
    elif len(results) > 0:
        #take first result
        return results[0]['uri']
    return None

def get_data(uri):
	#gets all data for a given uri
	query = " ".join([
		"SELECT DISTINCT (?r as ?relationship) (str(?p) as ?property)",
		"WHERE {",
		"<" + uri + "> ?r ?p.",
		"FILTER regex(?r,'dbpedia.org','i').",
		"FILTER(!isLiteral(?p) || lang(?p) = '' || langMatches(lang(?p), 'en'))",
		"}"
	])
	sparql.setQuery(query)
	results = sparql.query().convert()
	ret = {}
	for obj in results['results']['bindings']:
		ret[obj['relationship']['value']] = obj['property']['value']
	return ret

def get_links(uri):
	#gets links to other dbpedia entries for a given uri
	query = " ".join([
		"SELECT DISTINCT ?r1 ?p1 ?r2 ?p2 WHERE {{",
		"<http://dbpedia.org/resource/California> ?r1 ?p1.",
		"?p1 rdfs:label ?pl.",
		"FILTER regex(?r1,'dbpedia.org','i').",
		"}UNION{",
		"?p2 ?r2 <" + uri + ">.",
		"?p2 rdfs:label ?pl.",
		"FILTER regex(?r2,'dbpedia.org','i').",
		"}}"
	])
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
