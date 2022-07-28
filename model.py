import torch
import torch.nn as nn


class e_lstm_d(nn.Module):
    def __init__(self, num_nodes, historical_len, encoder_units, lstm_units, decoder_units):
        super(e_lstm_d, self).__init__()
        self.historical_len = historical_len
        self.num_nodes = num_nodes
        self.encoder_units = encoder_units
        self.stacked_lstm_units = lstm_units
        self.decoder_units = decoder_units

        self.encoder = nn.Sequential(
            nn.Linear(self.num_nodes*self.num_nodes, self.encoder_units[0]),
            nn.ReLU()
        )
        self.lstm = nn.Sequential(
            nn.LSTM(self.encoder_units[0], self.stacked_lstm_units[0], num_layers=2)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.stacked_lstm_units[0], self.decoder_units[0]*self.decoder_units[0]),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.decoder_units[0] , self.decoder_units[0]),
            nn.Sigmoid()
        )



    #LSTM只取最后一维
    def forward(self, inputs):

        x = inputs.view(-1, 10, self.num_nodes*self.num_nodes)
        x = self.encoder(x)
        x, h_n = self.lstm(x)
        x = torch.narrow(x,1,9,1)
        x = x.view(-1, 256)
        x = self.decoder1(x)
        x = x.view(-1, self.num_nodes, self.num_nodes)
        outputs = self.decoder2(x)

        return outputs

class ED_LSTM(nn.Module):
    def __init__(self, num_nodes, historical_len, encoder_units, lstm_units, decoder_units):
        super(ED_LSTM, self).__init__()
        self.historical_len = historical_len
        self.num_nodes = num_nodes
        self.encoder_units = encoder_units
        self.stacked_lstm_units = lstm_units
        self.decoder_units = decoder_units

        self.encoder = nn.Sequential(
            nn.Linear(self.num_nodes * self.num_nodes, self.encoder_units[0]),
            nn.ReLU()
        )
        self.lstm = nn.Sequential(
            nn.LSTM(self.encoder_units[0], self.stacked_lstm_units[0], num_layers=2, batch_first=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.stacked_lstm_units[0], self.decoder_units[0] * self.decoder_units[0]),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.decoder_units[0], self.decoder_units[0]),
            nn.Sigmoid()
        )

    # LSTM only takes the last dimension
    def forward(self, inputs):
        x = inputs.view(-1, self.historical_len, self.num_nodes * self.num_nodes)
        x = self.encoder(x)
        x, (h_n,c_n) = self.lstm(x)
        h_n = h_n[-1]

        x = self.decoder1(h_n)
        x = x.view(-1, self.num_nodes, self.num_nodes)
        outputs = self.decoder2(x)

        return outputs



if __name__== '__main__':
    import numpy as np
    import random


    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = ED_LSTM(num_nodes=274, historical_len=10, encoder_units=[128], lstm_units=[128, 256], decoder_units=[274])
    inputs = torch.randn(32, 10, 274, 274)

    output = model(inputs)




