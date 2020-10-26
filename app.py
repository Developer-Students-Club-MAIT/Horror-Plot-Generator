import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from flask import Flask, jsonify, request

app = Flask(__name__)
train_on_gpu = torch.cuda.is_available()

def stored_model(get_name):
    global model
    model = get_name

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
    answer = PlotGenerate(model, length, prime=start_words, top_k=5)
    return jsonify({'resultant_story': answer})

if __name__ == "__main__":
    from model_class import CharRNN
    model= torch.load('./model/model.pt', map_location=torch.device('cpu'))
    stored_model(model)
    app.run(debug=True)