from conceptnet5 import nodes
from conceptnet5 import query
from conceptnet5.uri import split_uri

from urllib.request import urlopen
from urllib.parse import quote
import json
import pickle

def get_results(feature):
    feature = feature.lower()
    ret = []
    with urlopen('http://conceptnet5.media.mit.edu/data/5.4/assoc' + quote('/c/en/'+feature) + "?filter=/c/en&limit=100") as response:
        html = response.read().decode('utf8')
        result = json.loads(html)
        for u,score in result["similar"]:
            s = float(score)
            if s >= 0.5:
                tmp = split_uri(u)
                if tmp[-1] != "neg":#ignore negative relationships
                    ret.append((tmp[2],1-s))#invert score priority
    return ret

import heapq

class PriorityQueue:
    #lower number == higher priority
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def push(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        return heapq.heappop(self.elements)[1]

def heuristic_cost(base,node):
    try:
        with urlopen('http://conceptnet5.media.mit.edu/data/5.4/assoc' + quote('/c/en/'+node) + "?filter=" + quote('/c/en/'+base) + "&limit=1") as response:
            html = response.read().decode('utf8')
            result = json.loads(html)
            return float(result["similar"][0][1])
    except:
        return 1


def a_star_search(graph, start, goal):

    if start not in graph or goal not in graph:
        return None

    frontier = PriorityQueue()
    frontier.push(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    found = False
    while not frontier.empty():
        current = frontier.pop()
        if current == goal:
            found = True
            break
        for n,s in graph[current]:
            new_cost = cost_so_far[current] + s
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                priority = new_cost + heuristic_cost(start,n)
                frontier.push(n, priority)
                came_from[n] = current
    if not found:
        return None

    path = [goal]
    tmp = came_from[goal]
    total_cost = 0
    while tmp:
        total_cost += cost_so_far[tmp]
        path.append(tmp)
        tmp = came_from[tmp]
    return path, total_cost



def construct_nodegraph(feature_set,bidirectional=True):
    nodegraph = {}
    explored = set()

    try:
        expanded_data = pickle.load(open("expanded_data.pkl", "rb"))
    except (OSError, IOError) as e:
        expanded_data = {}

    def explore(x,depth):
        if x in explored:
            return
        elif depth < 1:
            explored.add(x)
            tmp = nodegraph.setdefault(x,set())

            if x not in expanded_data:
                expanded_data[x] = get_results(x)

            for node, score in expanded_data[x]:
                if bidirectional:
                    nodegraph.setdefault(node,set()).add((x,score))
                tmp.add((node,score))
                explore(node,depth+1)

    feature_set = {nodes.standardized_concept_name('en',a) for x in feature_set for a in x.split()}

    for x in feature_set:
        explore(x,0)

    pickle.dump(expanded_data, open("expanded_data.pkl", "wb"))
    return nodegraph

def strongly_connected_components(graph):

        """yields sets of SCCs"""

        identified = set()
        stack = []
        index = {}
        lowlink = {}

        def dfs(v):
            index[v] = len(stack)
            stack.append(v)
            lowlink[v] = index[v]

            if not v in graph:
                return []

            for w,s in graph[v]:
                if w not in index:
                    yield from dfs(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w not in identified:
                    lowlink[v] = min(lowlink[v], lowlink[w])

            if lowlink[v] == index[v]:
                scc = set(stack[index[v]:])
                del stack[index[v]:]
                identified.update(scc)
                yield scc

        for v in graph:
            if v not in index:
                yield from dfs(v)


def connected_components(graph):
    seen = set()
    def component(node):
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            try:
                nodes |= {x[0] for x in graph[node]} - seen
            except:
                pass
            yield node

    for node in graph:
        if node not in seen:
            yield set(component(node))
