# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, n_units[0], bias=bias))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.Linear(n_units[i - 1], n_units[i], bias=bias))
        self.layer_list.append(nn.Linear(n_units[-1], output_dim, bias=bias))
        self.layer_num = layer_num + 1

    def forward(self, x):
        for i in range(self.layer_num):
            if i == self.layer_num-1:
                x = torch.sigmoid(self.layer_list[i](x))
            else:
                x = F.relu(self.layer_list[i](x))
        return x


# DynAE class
class DynAE(nn.Module):
    input_dim: int
    output_dim: int
    look_back: int
    bias: bool
    method_name: str
    encoder: MLP
    decoder: MLP

    def __init__(self, input_dim, output_dim, look_back,num_nodes, n_units=None, bias=True, **kwargs):
        super(DynAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.look_back = look_back
        self.bias = bias
        self.method_name = 'dynAE'
        print('model_name',self.method_name)
        self.num_nodes = num_nodes

        self.encoder = MLP(input_dim*self.look_back, output_dim, n_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1, 3)
        x = x.reshape(inputs.shape[0], inputs.shape[2], -1)
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return x_pred, hx


# L1 and L2 regularization loss
class RegularizationLoss(nn.Module):
    nu1: float
    nu2: float

    def __init__(self, nu1, nu2):
        super(RegularizationLoss, self).__init__()
        self.nu1 = nu1
        self.nu2 = nu2

    @staticmethod
    def get_weight(model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                # print('name: ', name)
                weight_list.append(weight)
        return weight_list

    def forward(self, model):
        loss = Variable(torch.FloatTensor([0.]), requires_grad=True).cuda() if torch.cuda.is_available() else Variable(
            torch.FloatTensor([0.]), requires_grad=True)
        # No L1 regularization and no L2 regularization
        if self.nu1 == 0. and self.nu2 == 0.:
            return loss
        # calculate L1-regularization loss and L2-regularization loss
        weight_list = self.get_weight(model)
        weight_num = len(weight_list)
        # print('weight num', weight_num)
        l1_reg_loss, l2_reg_loss = 0, 0
        for name, weight in weight_list:
            if self.nu1 > 0:
                l1_reg = torch.norm(weight, p=1)
                l1_reg_loss = l1_reg_loss + l1_reg
            if self.nu2 > 0:
                l2_reg = torch.norm(weight, p=2)
                l2_reg_loss = l2_reg_loss + l2_reg
        l1_loss = self.nu1 * l1_reg_loss / weight_num
        l2_loss = self.nu2 * l2_reg_loss / weight_num
        return l1_loss + l2_loss


# Loss used for DynAE, DynRNN, DynAERNN
class DynGraph2VecLoss(nn.Module):
    beta: float
    regularization: RegularizationLoss

    def __init__(self, beta, nu1, nu2):
        super(DynGraph2VecLoss, self).__init__()
        self.beta = beta
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        x_reconstruct, x_real, y_penalty = input_list[0], input_list[1], input_list[2]
        assert len(input_list) == 3
        reconstruct_loss = torch.mean(torch.sum(torch.square((x_reconstruct - x_real) * y_penalty), dim=1))
        regularization_loss = self.regularization(model)
        # print('total loss: ', main_loss.item(), ', reconst loss: ', reconstruct_loss.item(), ', L1 loss: ', l1_loss.item(), ', L2 loss: ', l2_loss.item())
        return reconstruct_loss + regularization_loss

# Loss used for DynAE, DynRNN, DynAERNN   re-define
class RE_DynGraph2VecLoss(nn.Module):
    beta: float
    regularization: RegularizationLoss

    def __init__(self, alpha):
        super(RE_DynGraph2VecLoss, self).__init__()
        self.alpha = alpha


    def forward(self,  y_true, y_pred):
        z = torch.ones_like(y_true)
        z = torch.add(z, torch.mul(y_true, self.alpha-1))
        reconstruct_loss = torch.mean(torch.sum(torch.pow((y_pred - y_true) * z,2), dim=1))

        return reconstruct_loss






