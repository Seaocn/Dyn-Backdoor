# coding: utf-8
from Baseline.dynAE import MLP
from Baseline.dynRNN import *

class MTMLP(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, look_back, bias=True):
        super(MTMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias

        self.layer_list = nn.ModuleList()
        for timestamp in range(look_back):
            self.layer_list.append(MLP(input_dim, output_dim, n_units, bias=bias))

    # x dim: [look_back,batch_size, input_dim]
    def forward(self, x):
        hx_list = []
        for timestamp in range(self.look_back):
            hx = self.layer_list[timestamp](x[timestamp,:, :])
            hx_list.append(hx)
        return torch.stack(hx_list, dim=0)


# DynAERNN class
class DynAERNN(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLLSTM
    decoder: MLLSTM

    def __init__(self, input_dim, output_dim, look_back,num_nodes,num_batch, ae_units=None, rnn_units=None, bias=True, **kwargs):
        super(DynAERNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'dynAERNN'
        print('model_name', self.method_name)
        self.num_nodes = num_nodes
        self.num_batch = num_batch

        self.ae_encoder = MTMLP(input_dim, output_dim, ae_units, look_back, bias=bias)
        self.rnn_encoder = ML_LSTM(output_dim, output_dim, rnn_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, ae_units[::-1], bias=bias)

    def forward(self, inputs):
        x = inputs.permute(1, 0, 2, 3)
        x = x.reshape(self.look_back, -1, self.num_nodes)
        ae_hx = self.ae_encoder(x)
        output, hx = self.rnn_encoder(ae_hx)
        # x_embedding
        x_embedding = hx.reshape(inputs.shape[0], inputs.shape[2], -1)

        x_pred = self.decoder(hx)
        x_pred = x_pred.reshape(inputs.shape[0], inputs.shape[2], self.num_nodes)


        return  x_pred,x_embedding
