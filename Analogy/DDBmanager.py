#! /usr/bin/env python3.5
from flask import Flask, request, render_template
import urllib
import urllib.request
import urllib.parse
import sys
import os
from pprint import pformat, pprint
import argparse

from domainDB import DomainManager

app = Flask(__name__)
app.root_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
DM = DomainManager("test.sqlite3", "./data files/managed/")

#print(DM.find_domains("USA",False))
#print(DM.find_domains("Canada",False))
#DM.reconcile_knowledge()
#DM.refresh_database()

@app.route('/')
def index():
    return "Hello"

@app.route('/test_query', methods=['GET'])
def get_features():
    q = request.args['query']
    print(q)
    rv = DM.find_domains(q,False)
    print(rv)
    return str(rv)

@app.route('/reconcile_db', methods=['GET'])
def reconcile_db():
    DM.reconcile_knowledge()
    return "ACK"

parser = argparse.ArgumentParser(description='Domain database manager.')
parser.add_argument('--host', nargs='?', const='127.0.0.1', help='the host address (default: 127.0.0.1)')
parser.add_argument('--port', nargs='?', const='5000', help='the port (default: 5000)')
parser.add_argument('--debug', action='store_true', help='Flask debug mode (default off)')
args = parser.parse_args()

app.run(host=args.host, port=args.port, debug=args.debug)