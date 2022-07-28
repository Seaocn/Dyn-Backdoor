# coding: utf-8
import torch.nn as nn


# DynRNN model and its components
# Multi-layer LSTM class
class MLLSTM(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(MLLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.LSTM(input_dim, n_units[0], bias=bias, batch_first=True))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.LSTM(n_units[i - 1], n_units[i], bias=bias, batch_first=True))
        self.layer_list.append(nn.LSTM(n_units[-1], output_dim, bias=bias, batch_first=True))
        self.layer_num = layer_num+1

    def forward(self, x):
        for i in range(self.layer_num):
            x, _ = self.layer_list[i](x)
        # return outputs and the last hidden embedding matrix
        return x, x[:, -1, :]


# Multi-layer LSTM class
class ML_LSTM(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(ML_LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.LSTM(input_dim, n_units[0], bias=bias, batch_first=False))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.LSTM(n_units[i - 1], n_units[i], bias=bias, batch_first=False))
        self.layer_list.append(nn.LSTM(n_units[-1], output_dim, bias=bias, batch_first=False))
        self.layer_num = layer_num+1


    def forward(self, x):
        for i in range(self.layer_num):
            #使用多块GPU
            # self.layer_list[i].flatten_parameters()
            x, _ = self.layer_list[i](x)
        # return outputs and the last hidden embedding matrix
        return x, x[-1,:, :]



# DynRNN class
class DynRNN(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLLSTM
    decoder: MLLSTM

    def __init__(self, input_dim, output_dim, look_back,num_nodes,num_batch, n_units=None, bias=True, **kwargs):
        super(DynRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'DynRNN'
        print('model_name', self.method_name)
        self.num_nodes = num_nodes
        self.num_batch = num_batch

        self.encoder = ML_LSTM(input_dim, output_dim, n_units, bias=bias)
        self.decoder = ML_LSTM(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, inputs):
        # 交换维度
        x = inputs.permute(1, 0, 2, 3)
        x = x.reshape(self.look_back, -1, self.num_nodes)
        output, hx = self.encoder(x)
        x_embedding = hx.reshape(inputs.shape[0], inputs.shape[2], -1)
        _, x_pred = self.decoder(output)
        x_pred = x_pred.reshape(inputs.shape[0], inputs.shape[2], self.num_nodes)

        return x_pred, x_embedding



if __name__== '__main__':

    import torch
    import torch.nn as nn
    inputs = torch.randn(32, 10, 20, 10)
