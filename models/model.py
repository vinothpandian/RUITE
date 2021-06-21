import torch.nn as nn
from config.uiconfig import UIconfig
import math
import torch

class Embedder(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedder, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# copied from the the annotated transformer (Implementation in Pytorch by Harvard Uni group of
# the orignal implementation by Viswani et. al in TensorFlow)
class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=UIconfig.seq_len):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# copied from the the annotated transformer (Implementation in Pytorch by Harvard Uni group of
# the orignal implementation by Viswani et. al in TensorFlow)
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class classificationHead(nn.Module):
    def __init__(self, hidden_out , vocab_size , drop_out):
        super().__init__()
        self.hidden = hidden_out
        self.drop_out = nn.Dropout(p=drop_out)
        self.relu = nn.ReLU()
        # map output to vocab size:
        self.prediction_layer = nn.Linear(self.hidden, vocab_size)
    def forward(self, x):
        #x = self.drop_out(x)
        x = self.prediction_layer(x)
        return x

# Using PyTorch Transformer encoder class
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, drop_out=0.2):
        super().__init__()

        self.embed = Embedder(d_model, vocab_size)
        # self.norm = Norm(d_model)
        self.pose = PositionalEncoder(d_model, drop_out)
        #Single layer of encoder:
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=64,
                                                        dropout=drop_out)
        #Replicate encoder layer N times
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=N)
        #Using output of Tranf for classifcation:
        self.classification_head = classificationHead(d_model, vocab_size, drop_out)

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pose(x)
        # transpose because we want output in batch first format
        e_output = self.encoder(x, src_key_padding_mask=mask).transpose(0, 1)
        out_cls = self.classification_head(e_output)
        return out_cls

