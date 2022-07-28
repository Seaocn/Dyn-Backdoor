import torch.nn as nn
import torch




class Generator(nn.Module):
    def __init__(self, intput_size, hidden_size):#input_size [args.num_nodes]  hidden_size[128, args.num_nodes]
        super(Generator, self).__init__()
        self.input_size = intput_size
        self.hidden_size = hidden_size


        self.gconv1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size[0]),
            torch.nn.ReLU()
        )
        self.gconv2 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            torch.nn.Sigmoid()
        )

    def forward(self, Adjacency_Modified):
        H_1 = self.gconv1(Adjacency_Modified)
        H_2 = self.gconv2(H_1)



        return H_2




class Generator_batch(nn.Module):
    def __init__(self, num_nodes, historical_len, encoder_units, lstm_units, decoder_units):#input_size [args.num_nodes]  hidden_size[128, args.num_nodes]
        super(Generator_batch, self).__init__()
        self.historical_len = historical_len
        self.num_nodes = num_nodes
        self.encoder_units = encoder_units
        self.stacked_lstm_units = lstm_units
        self.decoder_units = decoder_units

        self.encoder = nn.Sequential(
            nn.Linear(self.num_nodes, self.encoder_units[0]),
            nn.ReLU()
        )
        self.lstm = nn.Sequential(
            nn.LSTM(self.encoder_units[0], self.stacked_lstm_units[0], num_layers=2, batch_first=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.stacked_lstm_units[0], self.decoder_units[0]),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = inputs.view(-1, self.historical_len, self.num_nodes)
        x = self.encoder(x)
        x, (h_n, c_n) = self.lstm(x)
        x = self.decoder1(x)
        x = x.view(-1, self.historical_len, 1, self.num_nodes)


        return x



class Generator_tanh(nn.Module):
    def __init__(self, num_nodes, historical_len, encoder_units, lstm_units, decoder_units):#input_size [args.num_nodes]  hidden_size[128, args.num_nodes]
        super(Generator_tanh, self).__init__()
        self.historical_len = historical_len
        self.num_nodes = num_nodes
        self.encoder_units = encoder_units
        self.stacked_lstm_units = lstm_units
        self.decoder_units = decoder_units

        self.encoder = nn.Sequential(
            nn.Linear(self.num_nodes, self.encoder_units[0]),
            nn.ReLU()
        )
        self.lstm = nn.Sequential(
            nn.LSTM(self.encoder_units[0], self.stacked_lstm_units[0], num_layers=2, batch_first=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.stacked_lstm_units[0], self.decoder_units[0]),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = inputs.view(-1, self.historical_len, self.num_nodes)
        x = self.encoder(x)
        x, (h_n, c_n) = self.lstm(x)
        x = self.decoder1(x)
        x = x.view(-1, self.historical_len, 1, self.num_nodes)


        return x


