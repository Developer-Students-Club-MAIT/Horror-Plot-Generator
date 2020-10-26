import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from flask import Flask, jsonify, request

global model

class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,  
                               drop_prob=0.5, lr=0.001, *args, **kwargs):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, len(self.chars))
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        # pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        
        out = self.fc(out)
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

train_on_gpu = torch.cuda.is_available()
app = Flask(__name__)

@app.route('/api', methods=['GET'])
def home():
    length = 100
    start_words = 'the'
    query = request.args
    model= torch.load('./model/model.pt', map_location=torch.device('cpu'))
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
    model= torch.load('./model/model.pt', map_location=torch.device('cpu'))
    app.run(debug=True)