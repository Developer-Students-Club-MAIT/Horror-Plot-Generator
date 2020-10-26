import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from flask import Flask, jsonify, request
from model_class import CharRNN

app = Flask(__name__)
train_on_gpu = torch.cuda.is_available()
app.config['model']= torch.load('./model/model.pt', map_location=torch.device('cpu'))

@app.route('/api', methods=['GET'])
def home():
    length = 100
    start_words = 'the'
    query = request.args
    plot_length = query.get('len_choice')
    if plot_length == 'small':
        length = 500
    elif plot_length == 'medium':
        length = 1200
    elif plot_length == 'large':
        length = 2000
    start_words = query.get('start_words')
    answer = PlotGenerate(app.config['model'], length, prime=start_words, top_k=5)
    return jsonify({'resultant_story': answer})

if __name__ == "__main__":
    app.run(debug=True)