import torch
import torch.nn as nn
from torchvision import transforms as T



class E_LSTM_D(nn.Module):
    def __init__(self, num_nodes, historical_len, encoder_units, lstm_units, decoder_units):
        super(E_LSTM_D, self).__init__()
        self.historical_len = historical_len
        self.num_nodes = num_nodes
        self.encoder_units = encoder_units
        self.stacked_lstm_units = lstm_units
        self.decoder_units = decoder_units
        self.method_name = 'E_LSTM_D'
        print('model_name', self.method_name)

        self.encoder = nn.Sequential(
            nn.Linear(self.num_nodes, self.encoder_units[0]),
            nn.ReLU()
        )
        self.lstm1 = nn.Sequential(
            nn.LSTM(self.encoder_units[0], self.stacked_lstm_units[0],batch_first=True)
        )
        self.lstm2 = nn.Sequential(
            nn.LSTM(self.stacked_lstm_units[0], self.stacked_lstm_units[1],batch_first=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.stacked_lstm_units[1], self.decoder_units[0]),
            nn.Sigmoid()
        )


    def forward(self, inputs):
        x = self.encoder(inputs)
        x = x.view(inputs.shape[0], self.historical_len, -1)
        x = T.Lambda(lambda x: torch.sum(x, axis=1))(x)
        x = x.view(inputs.shape[0], inputs.shape[2], self.encoder_units[0])
        x, h_n = self.lstm1(x)
        x, h_n = self.lstm2(x)
        outputs = self.decoder(x)

        return outputs, x