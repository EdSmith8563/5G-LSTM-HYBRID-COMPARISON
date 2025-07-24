import torch.nn as nn

def shared_bidirectional_lstm(input_dim, hidden_dim):
    return nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        batch_first=True,
        bidirectional=True,
        num_layers=2,
        dropout=0.2
    )
