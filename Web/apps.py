from flask import Flask
from flask import render_template
from flask import request
import xml.etree.ElementTree as ET
import os
import pickle, json
import nltk
import difflib
import re
import json
from markupsafe import Markup
from torch.utils.data.dataloader import DataLoader
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    with open('../Data/depressive_doc.pkl', 'rb')as fpick:
        Depression = pickle.load(fpick)
    with open('../Data/bipolar_doc.pkl', 'rb')as fpick:
        Bipolar = pickle.load(fpick)
    return render_template('index.html', Depression = Depression, Bipolar = Bipolar)

@app.route('/depressive')
def depressive():
    with open('../Data/res_depressive_doc.pkl', 'rb')as fpick:
        Depression = pickle.load(fpick)
    return render_template('depressive.html', Depression = Depression)

@app.route('/bipolar')
def bipolar():
    with open('../Data/res_bipolar_doc.pkl', 'rb')as fpick:
        Bipolar = pickle.load(fpick)
    return render_template('bipolar.html', Bipolar = Bipolar)

@app.route('/compare')
def compart():
    with open('../Data/imporant_bipolar_doc.pkl', 'rb')as fpick:
        imp_Bipolar = pickle.load(fpick)
    with open('../Data/imporant_depressive_doc.pkl', 'rb')as fpick:
        imp_Depression = pickle.load(fpick)
    with open('../Data/res_bipolar_doc.pkl', 'rb')as fpick:
        Bipolar = pickle.load(fpick)
    with open('../Data/res_depressive_doc.pkl', 'rb')as fpick:
        Depression = pickle.load(fpick)
    return render_template('compare.html', Depression = Depression, Bipolar = Bipolar, imp_Depression = imp_Depression, imp_Bipolar = imp_Bipolar)

if __name__ == '__main__':
    app.debug = True
    # app.run()
    app.run(host='0.0.0.0', port=5001)