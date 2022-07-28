# coding: utf-8
import torch
import torch.nn as nn
from config import args

class DDNE(nn.Module):
    def __init__(self,enc_hidden_dim,dec_hidden_dim,num_nodes,historical_len):
        super(DDNE, self).__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.num_nodes = num_nodes
        self.historical_len = historical_len
        self.method_name = 'DDNE'
        print('model_name', self.method_name)
        self.encoder = nn.GRU(input_size=self.num_nodes,hidden_size=self.enc_hidden_dim,num_layers=1,batch_first=False)
        self.decoder = nn.Sequential(
            nn.Linear(self.enc_hidden_dim*self.historical_len,self.dec_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.dec_hidden_dim,self.num_nodes),
            nn.Sigmoid()
        )


    def forward(self,inputs):
        x = inputs.permute(1, 0, 2, 3)
        x = x.reshape(self.historical_len, -1, self.num_nodes)
        output, hr = self.encoder(x)
        output = output.permute(1, 0, 2)
        output = output.reshape(-1, self.enc_hidden_dim * self.historical_len)
        # embedding
        x_embeding = output.reshape(inputs.shape[0], inputs.shape[2], -1)
        x_pred = self.decoder(output)
        x_pred = x_pred.reshape(-1, inputs.shape[2], inputs.shape[3])

        return x_pred,x_embeding



if __name__ == '__main__':
    net = DDNE(128,128,274,10)   #The decoder consists of two layers with 128,151 neurons respectively
    inputs = torch.randn(5, 10, 274, 274)
    output = net(inputs)
    print(output.shape)
