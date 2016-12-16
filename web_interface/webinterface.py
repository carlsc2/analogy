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
from pprint import pformat
from analogy_a1 import AIMind
import os.path

cache = {}

#have to do this because of flask nonsense
def cache_load(f):
    try:
        return cache[f]
    except:
        cache[f] = AIMind(filename=f)
        return cache[f]

data_dir = "./data files/"
def full_filename(filename):
    #add path to filename
    return os.path.join(data_dir, filename)

app = Flask(__name__)
app.root_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

@app.route('/')
def index():
    return render_template('index.html', file_list=list_files())

@app.route('/get_analogy', methods=['POST'])
def get_analogy():
    #print("file1: ", request.form['file1'])
    #print("file2: ", request.form['file2'])
    #print("feature1: ", request.form['feature1'])
    #print("feature2: ", request.form['feature2'])
    return json.dumps(cache_load(request.form['file1']).get_analogy(request.form['feature1'],
                                                                    request.form['feature2'],
                                                                    cache_load(request.form['file2'])))


@app.route('/print_analogy', methods=['POST'])
def print_analogy():
    return pformat(cache_load(request.form['file1']).get_analogy(request.form['feature1'],
                                                        request.form['feature2'],
                                                        cache_load(request.form['file2'])), indent=4, width=80)


@app.route('/find_best_analogy', methods=['POST'])
def find_best_analogy():
    return pformat(cache_load(request.form['file1']).find_best_analogy(request.form['feature'],
                                                                       cache_load(request.form['file2'])),
                   indent=4, width=300)

def list_files():
    return [f[13:] for f in glob.glob('./data files/*.xml')]

@app.route('/get_features', methods=['POST'])
def get_features():
    f = request.form['file']
    print("get_features: ", f)
    return json.dumps(list(cache_load(f).features.keys()))

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
        cache[f] = AIMind(filename=filename)
    except:
        return "Invalid file format"
    return "File added"
    

parser = argparse.ArgumentParser(description='Analogy web interface server.')
parser.add_argument('--host', nargs='?', const='127.0.0.1', help='the host address (default: 127.0.0.1)')
parser.add_argument('--port', nargs='?', const='5000', help='the port (default: 5000)')
parser.add_argument('--debug', action='store_true', help='Flask debug mode (default off)')
args = parser.parse_args()
for f in list_files():
    fname = full_filename(f)
    try:
        cache[f] = AIMind(filename=fname)
    except:
        print("file %s is corrupt, ignoring"%fname)
        os.rename(fname, fname + "broken")
        
print("Files loaded")
for key in cache.keys():
    print("\t%s"%key)
app.run(host=args.host, port=args.port, debug=args.debug)
