import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from flask import Flask, jsonify, request, render_template
from model_class import CharRNN

app = Flask(__name__)
train_on_gpu = torch.cuda.is_available()
model = get_model()

@app.route('/')
def home():
    return render_template('index.html', choices= [{'name': 'small'}, {'name': 'medium'}, {'name': 'large'}], text='')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    length = 100
    start_words = 'the'
    plot_length = 'small'
    if request.method == 'POST':
        plot_length = request.form.get('model-select')
        start_words = request.form.get('ip')
    if plot_length == 'small':
        length = 500
    elif plot_length == 'medium':
        length = 1200
    elif plot_length == 'large':
        length = 2000   
    start_words = start_words.lower()
    answer = PlotGenerate(model, length, prime=start_words, top_k=5)
    return render_template('index.html',choices= [{'name': 'small'}, {'name': 'medium'}, {'name': 'large'}], text=answer)


@app.route('/api', methods=['GET'])
def predict_api():
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
    app.run(debug=True)