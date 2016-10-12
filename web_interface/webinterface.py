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
from analogy import AIMind

cache = {}



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', file_list=list_files())

@app.route('/get_analogy', methods=['POST'])
def get_analogy():
    return pformat(cache[request.form['file1']].get_analogy(request.form['feature1'],
                                                        request.form['feature2'],
                                                        cache[request.form['file2']]), indent=4, width=300)

@app.route('/find_best_analogy', methods=['POST'])
def find_best_analogy():
    return pformat(cache[request.form['file1']].find_best_analogy(request.form['feature'], cache[request.form['file2']]), indent=4, width=300)

def list_files():
    return [f[13:] for f in glob.glob('./data files/*.xml')]

@app.route('/get_features', methods=['POST'])
def get_features():
    f = request.form['file']
    return json.dumps(list(cache[f].features.keys()))
    


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analogy web interface server.')
    parser.add_argument('--host', nargs='?', const='127.0.0.1', help='the host address (default: 127.0.0.1)')
    parser.add_argument('--port', nargs='?', const='5000', help='the port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Flask debug mode (default off)')
    args = parser.parse_args()
    for f in list_files():
        cache[f] = AIMind(filename='./data files/'+f)
    print("Files loaded")
    for key in cache.keys():
        print("\t%s"%key)
    app.run(host=args.host, port=args.port, debug=args.debug)
