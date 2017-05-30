#! /usr/bin/env python3
from flask import Flask, request, render_template
import urllib
import urllib.request
import urllib.parse
import json
import cgi
import glob
import sys
import argparse
from pprint import pformat, pprint
from ..core import *
from ..utils.utils import AIMind, DomainLoader
import os.path

cache = {}
allow_file_write = False

DATADIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "../data files/"))
def full_filename(filename):
    #add path to filename
    return os.path.join(DATADIR, filename)

#have to do this because of flask nonsense
def cache_load(f):
    fname = full_filename(f)
    try:
        return cache[f]
    except KeyError:
        try:
            cache[f] = DomainLoader(filename=fname, cachefile=fname+"_cache.pkl").domain
        except:
            try:
                cache[f] = AIMind(filename=fname, cachefile=fname+"_cache.pkl").domain
            except Exception as e:
                print(e)
                print("cannot load file %s, ignoring"%fname)

        return cache[f]

app = Flask(__name__)
app.root_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

def clean(x, form):
    if x is None:
        return "No analogy could be made."
    if form.get('sanitize') == "true":
        x = x.replace("<","&lt;")
        x = x.replace(">","&gt;")
    if form.get('clear') == "true":
        x = x.replace("<","")
        x = x.replace(">","")
    return x


@app.route('/')
def index():
    return render_template('index.html', file_list=list_files())

@app.route('/get_analogy', methods=['POST'])
def get_analogy():
    cmode = 3 if request.form.get('cluster') == 'true' else 0
    return json.dumps(analogy.make_analogy(request.form['concept1'],
                                           cache_load(request.form['file1']),
                                           request.form['concept2'],
                                           cache_load(request.form['file2']),
                                           cluster_mode=cmode))

@app.route('/get_analogy_explain', methods=['POST'])
def get_analogy_explain():
    cmode = 3 if request.form.get('cluster') == 'true' else 0
    result = analogy.make_analogy(request.form['concept1'],
                                  cache_load(request.form['file1']),
                                  request.form['concept2'],
                                  cache_load(request.form['file2']),
                                  cluster_mode=cmode)

    explanation = analogy.explain_analogy(result)
    
    if result != None:
        return json.dumps({"explanation":clean(explanation, request.form),
                           "error":""})
    else:
        return json.dumps({"explanation":"",
                           "error":"No analogy could be made."})

@app.route('/find_best_analogy', methods=['POST'])
def find_best_analogy():
    cmode = 4 if request.form.get('cluster') == 'true' else 0
    knn = request.form.get('knn')
    try:
        knn = int(knn)
    except:
        knn = None
    return json.dumps(analogy.find_best_analogy(request.form['concept'],
                                                cache_load(request.form['file1']),
                                                cache_load(request.form['file2']),
                                                cluster_mode=cmode,
                                                knn_filter=knn))

@app.route('/find_best_analogy_explain', methods=['POST'])
def find_best_analogy_explain():
    cmode = 4 if request.form.get('cluster') == 'true' else 0
    knn = request.form.get('knn')
    try:
        knn = int(knn)
    except:
        knn = None
    result = analogy.find_best_analogy(request.form['concept'],
                                       cache_load(request.form['file1']),
                                       cache_load(request.form['file2']),
                                       knn_filter=knn,
                                       cluster_mode=cmode)
    explanation = analogy.explain_analogy(result)

    if result != None:
        return json.dumps({"analogy":clean(explanation, request.form),
                           "error":""})
    else:
        return json.dumps({"analogy":"",
                           "error":"No analogy could be made."})

@app.route('/print_analogy', methods=['POST'])
def print_analogy():
    cmode = 3 if request.form.get('cluster') == 'true' else 0
    result = analogy.make_analogy(request.form['concept1'],
                                  cache_load(request.form['file1']),
                                  request.form['concept2'],
                                  cache_load(request.form['file2']),
                                  cluster_mode=cmode)

    x = {}
    x["explanation"] = clean(analogy.explain_analogy(result), request.form)
    x["analogy"] = clean(pformat(result, indent=4, width=80), request.form)
    return json.dumps(x)

@app.route('/print_best_analogy', methods=['POST'])
def print_best_analogy():
    cmode = 4 if request.form.get('cluster') == 'true' else 0
    knn = request.form.get('knn')
    try:
        knn = int(knn)
    except:
        knn = None
    print("knn: ",knn)
    result = analogy.find_best_analogy(request.form['concept'],
                                       cache_load(request.form['file1']),
                                       cache_load(request.form['file2']),
                                       knn_filter=knn,
                                       cluster_mode=cmode)

    x = {}
    x["explanation"] = clean(analogy.explain_analogy(result), request.form)
    x["analogy"] = clean(pformat(result, indent=4, width=80), request.form)
    return json.dumps(x)

def list_files():
    return [os.path.basename(f) for f in glob.glob(DATADIR+'/*.xml')] + \
           [os.path.basename(f) for f in glob.glob(DATADIR+'/*.json')]

@app.route('/get_concepts', methods=['POST'])
def get_concepts():
    f = request.form['file']
    print("get_concepts: ", f)
    return json.dumps(list(cache_load(f).nodes.keys()))

@app.route('/check_file', methods=['POST'])
def check_file():
    #check if a file exists
    filename = full_filename(request.form['file'])
    if os.path.isfile(filename):
        return "true"
    else:
        return "false"

@app.route('/add_file', methods=['POST'])
def add_file():
    if allow_file_write:
        data = request.get_json()
        filename = full_filename(data['file'])
        
        #filename = full_filename(request.form['file'])
        print("adding file: ",filename)
        #add a file if it doesn't already exist
        if os.path.isfile(filename):
            if not data['override'] == "true":
                print("file %s already exists"%filename)
                return "File already exists"
        with open(filename,"wb+") as f:
            f.write(data["data"].encode("utf-8"))
        try:
            cache_load(filename)
        except:
            return "Invalid file format"
        return "File added"
    else:
        return "File write disabled"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analogy web interface server.')
    parser.add_argument('--host', nargs='?', const='127.0.0.1', help='the host address (default: 127.0.0.1)')
    parser.add_argument('--port', nargs='?', const='5000', help='the port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Flask debug mode (default off)')
    parser.add_argument('--write_file', action='store_true', help='Enable file writing (default off)')
    parser.add_argument('--data_dir', nargs='?', help='location of data files (defaults to package)')
    args = parser.parse_args()

    if args.data_dir != None:
        DATADIR = args.data_dir

    print("Data file location: ", DATADIR)

    allow_file_write = args.write_file
    if(allow_file_write):
        print("File writing ENABLED")
    else:
        print("File writing DISABLED")

    #load files from domain
    for f in list_files():
        fname = full_filename(f)
        print("loading %s..."%fname)
        cache_load(f)
            
    print("Files loaded:")
    for key in cache.keys():
        print("\t%s"%key)
    app.run(host=args.host, port=args.port, debug=args.debug)
