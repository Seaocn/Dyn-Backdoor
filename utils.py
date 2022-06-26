import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from config import args, device
import copy
import random
from backdoor import sub_replace


class Mydatasets(Dataset):
    def __init__(self, X,Y):
        super(Mydatasets, self).__init__()
        self.x = X
        self.y = Y
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return self.len

# class Mydatasets(Dataset):
#     def __init__(self, X,Y):
#         super(Mydatasets, self).__init__()
#         self.x = X
#         self.y = Y
#         self.len = self.x.shape[0]
#
#     def __getitem__(self, index):
#         return torch.tensor(self.x[index]).to(device), torch.tensor(self.y[index]).to(device)
#
#
#     def __len__(self):
#         return self.len


def generate_batch(X, Y, testX, testY, batch_size, sub_list,trainX_idx,train_edge_idx,isTrain=True):
    if isTrain:
        np.random.seed(120)
        np.random.shuffle(X)
        np.random.seed(120)
        np.random.shuffle(Y)
    num_batches = len(X) // batch_size  #取整
    for i in range(num_batches):
        input_X = torch.FloatTensor(X[i:i+args.batch_size])
        input_Y = torch.FloatTensor(Y[i:i+args.batch_size])
        test_X  = torch.FloatTensor(testX[i:i+args.batch_size])
        test_Y = torch.FloatTensor(testY[i:i + args.batch_size])
        #加入假节点需要优化的子图结构
        input_X, input_Y = sub_replace(input_X,input_Y,sub_list,trainX_idx,train_edge_idx)

        yield input_X.to(device), input_Y.to(device), test_X.to(device),test_Y.to(device)



#
def get_auc(x, y):
    return roc_auc_score(np.reshape(y, (-1, )), np.reshape(x, (-1, )))
#
#
def get_err_rate(x, y):
    return np.sum(np.abs(x - y)) / np.sum(y)


def evaluate(test_pred_X, ture_Y):
    template = np.ones((args.num_nodes, args.num_nodes)) - np.identity(args.num_nodes)
    aucs, err_rates = [], []
    for i in range(test_pred_X.shape[0]):
        x_pred = test_pred_X[i] * template
        er_pred_X = np.where(test_pred_X[i]>=0.5, 1,0)
        aucs.append(get_auc(np.reshape(x_pred, (-1,)), np.reshape(ture_Y[i], (-1,))))
        err_rates.append(get_err_rate(er_pred_X, ture_Y[i]))
    return aucs, err_rates
#
#
# def load_data(filePath):
#     if not os.path.exists(filePath):
#         raise FileNotFoundError
#     else:
#         return np.load(filePath)
#
#


# def refined_loss(y_true, y_pred, beta):
#     weight = y_true * (beta - 1) + 1  #这个就是惩罚矩阵，对存在连边出现错误，惩罚力度更大
#     loss = K.mean(K.sum(tf.multiply(weight, K.square(y_true - y_pred)), axis=1), axis=-1)
#     return loss



#因为tensor的问题，导致一些操作无法进行
# def refined_loss(y_true, y_pred,beta):
#     weight = y_true * (beta - 1) + 1  #这个就是惩罚矩阵，对存在连边出现错误，惩罚力度更大
#     loss = torch.mean(torch.sum(torch.mul(weight, torch.pow((y_true-y_pred),2)), dim=1))
#     return loss


def build_refined_loss(beta):
    def refined_loss(y_true, y_pred):
        weight = y_true * (beta - 1) + 1  #这个就是惩罚矩阵，对存在连边出现错误，惩罚力度更大
        return torch.mean(torch.sum(torch.mul(weight, torch.pow((y_true-y_pred),2)), dim=1))
        # return torch.sum(torch.sum(torch.mul(weight, torch.pow((y_true - y_pred), 2)), dim=1))
    return refined_loss

def build_tri_loss(beta):
    def refined_tri_loss(y_true, y_pred):
        weight = y_true * (beta - 1) + 1  #这个就是惩罚矩阵，对存在连边出现错误，惩罚力度更大
        return torch.sum(torch.sum(torch.mul(weight, torch.pow((y_true-y_pred),2)), dim=1))
        # return torch.sum(torch.sum(torch.mul(weight, torch.pow((y_true - y_pred), 2)), dim=1))
    return refined_tri_loss



#触发器生成器
def build_G_loss(beta,target_list):
    def refined_G_loss(y_true, y_pred):
        mask = torch.zeros_like(y_true)

        mask[:,target_list[0][1]:target_list[0][1]+1,:] = 1
        weight = y_true + 1  #这个就是惩罚矩阵，对存在连边出现错误，惩罚力度更大
        weight[:,target_list[0][1]:target_list[0][1]+1,target_list[0][2]] = beta

        return torch.sum(torch.sum(torch.mul(weight, torch.pow(torch.mul(torch.sub(y_true, y_pred), mask), 2)), dim=1))
        # return torch.sum(torch.sum(torch.mul(weight, torch.pow((y_true - y_pred), 2)), dim=1))
    return refined_G_loss




class StructuralLoss(nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        super(StructuralLoss, self).__init__()

    def forward(self, y_true, y_pred):
        z = torch.ones_like(y_true)
        z = torch.add(z, torch.mul(y_true, self.alpha))
        return nn.BCELoss(weight=z,size_average=True)(y_pred, y_true)




class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()

    def forward(self, y_true, y_pred, target_list):
        mask = torch.zeros_like(y_true)
        for i in range(mask.shape[0]):
            mask[i,target_list[0][1],target_list[0][2]] = 1
        # mask_idx = (torch.LongTensor([0]),torch.LongTensor([target_list[0][1]]), torch.LongTensor([target_list[0][2]]))
        # mask = mask.index_put_(mask_idx, torch.cuda.FloatTensor([1]))

        # if mask[0,target_list[0][0],target_list[0][1]] == 1:
        #     print('***')

        # return -nn.BCELoss(weight=mask)(y_pred, y_true)
        return torch.sum(torch.sum(torch.pow(torch.mul(torch.sub(y_true, y_pred), mask), 2), dim=1))

#全局损失函数
class All_net_MaskedLoss(nn.Module):
    def __init__(self):
        super(All_net_MaskedLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.sum(torch.pow(torch.sub(y_true, y_pred), 2))

class All_net_MaskedLoss_Y(nn.Module):
    def __init__(self):
        super(All_net_MaskedLoss_Y,self).__init__()

    def forward(self, y_true, y_pred):
        return -torch.sum(torch.pow(torch.sub(y_true, y_pred), 2))

#找到频繁出现的连边
def find_most_edges(trainY):
    edges = []
    for i in range(30):
        for row in range(278):
            for col in range(278):
                if trainY[i][row][col] == 1:
                    edges.append((row,col))

    # 计算出现该边的个数
    # a = {}
    most_edges = []
    for j in edges:
        # a[j] = edges.count(j)
        if edges.count(j)>=15:
            most_edges.append(j)
    print(most_edges)


def fast_gradient_attack(gradients, inputs):
    #单边
    # _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[:,target_list[0][0],:]), (-1,)),descending=True)
    # ts, idx = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
    _inputs = copy.deepcopy(inputs)
    attack_num = 0
    modify_link = 0


    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients), (-1,)),
                                 descending=True)
    while(1):
        #全局连边
        # ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        # idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
        #             sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        # g = gradients[ts, idx_X, idx_Y]   #连边梯度
        # v = inputs[0, ts, idx_X, idx_Y]   #连边本身的状态
        #直接连边
        ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
                sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        g = gradients[ts, idx_X, idx_Y]  # 连边梯度
        v = inputs[0, ts, idx_X, idx_Y]

        # value = 1 if v == 0 else 0
        value = is_to_modify(g, v)
        if value != -1 :
            _inputs = _inputs.index_put_((torch.LongTensor([0]), torch.LongTensor([ts]),
                                          torch.LongTensor([idx_X]), torch.LongTensor([idx_Y])),torch.cuda.FloatTensor([value]))
            modify_link += 1
            attack_num += 1
            # break
        else:
            attack_num += 1


        if attack_num >= int(len(sorted_index)*0.01) :
            eixt_FGA = None
            break


    return _inputs, eixt_FGA


def times_fast_gradient_attack(gradients, inputs,attack_num):
    #单边
    # _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[:,target_list[0][0],:]), (-1,)),descending=True)
    # ts, idx = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
    _inputs = copy.deepcopy(inputs)
    modify_link = 0
    eixt_FGA = True


    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients), (-1,)),
                                 descending=True)
    while(1):
        #全局连边
        # ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        # idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
        #             sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        # g = gradients[ts, idx_X, idx_Y]   #连边梯度
        # v = inputs[0, ts, idx_X, idx_Y]   #连边本身的状态
        #直接连边
        ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
                sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        g = gradients[0,ts, idx_X, idx_Y]  # 连边梯度
        v = inputs[0, ts, idx_X, idx_Y]

        # value = 1 if v == 0 else 0
        value = is_to_modify(g, v)
        if value != -1 :
            _inputs = _inputs.index_put_((torch.LongTensor([0]), torch.LongTensor([ts]),
                                          torch.LongTensor([idx_X]), torch.LongTensor([idx_Y])),torch.cuda.FloatTensor([value]))
            modify_link += 1
            attack_num += 1
            break
        else:
            attack_num += 1


        if attack_num >= int(args.num_nodes*10*0.9) :
            eixt_FGA = None
            break


    return _inputs, eixt_FGA, attack_num

def fast_gradient_attack_ALL(gradients, inputs):
    #单边
    # _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[:,target_list[0][0],:]), (-1,)),descending=True)
    # ts, idx = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
    _inputs = copy.deepcopy(inputs)
    attack_num = 0
    modify_link = 0

    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients), (-1,)),
                                 descending=True)
    while(1):
        #全局连边
        ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
                    sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        g = gradients[ts, idx_X, idx_Y]   #连边梯度
        v = inputs[0, ts, idx_X, idx_Y]   #连边本身的状态

        value = is_to_modify(g, v)
        if value != -1 :
            _inputs = _inputs.index_put_((torch.LongTensor([0]), torch.LongTensor([ts]),
                                          torch.LongTensor([idx_X]), torch.LongTensor([idx_Y])),torch.cuda.FloatTensor([value]))
            modify_link += 1
            attack_num += 1
        else:
            attack_num += 1


        if attack_num >= int(len(sorted_index)*0.9) or modify_link >= 167*167*10*0.5:
            break
        # if attack_num >= int(len(sorted_index)*0.9):
        #     break


    return _inputs



def fast_gradient_attack_ALL_Y(gradients, inputs):
    #单边
    # _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[:,target_list[0][0],:]), (-1,)),descending=True)
    # ts, idx = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
    _inputs = copy.deepcopy(inputs)
    attack_num = 0
    modify_link = 0

    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients), (-1,)),
                                 descending=True)
    while(1):
        #全局连边

        idx_X, idx_Y = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
        g = gradients[idx_X, idx_Y]   #连边梯度
        v = inputs[0, idx_X, idx_Y]   #连边本身的状态

        value = is_to_modify(g, v)
        if value != -1 :
            _inputs = _inputs.index_put_((torch.LongTensor([0]),
                                          torch.LongTensor([idx_X]), torch.LongTensor([idx_Y])),torch.cuda.FloatTensor([value]))
            modify_link += 1
            attack_num += 1
        else:
            attack_num += 1


        if attack_num >= int(len(sorted_index)*0.9) or modify_link >= 274*274*0.5:
            break
        # if attack_num >= int(len(sorted_index)*0.9):
        #     break


    return _inputs



def one_step_attack(gradients, inputs, target_list):
    # gradients: (historical_len, num_nodes)
    # inputs: (historical_len, 1, ,1, num_nodes)
    # max_grad = torch.FloatTensor([torch.max(torch.abs(gradients[ts]))])
    # min_grad = torch.FloatTensor([torch.min(torch.abs(gradients[ts]))])
    # rescaled_grad = max_grad - gradients / max_grad - min_grad
    _inputs = copy.deepcopy(inputs)
    attack_num = 0
    value = -1
    IS_posion = True
    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[:, target_list[0][0], :] * 1000), (-1,)),
                                 descending=True)
    while(value==-1):
        ts, idx = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
        g = gradients[ts, target_list[0][0], idx]  # 连边梯度
        v = inputs[0, ts, target_list[0][0], idx]  # 连边本身的状态
        value = is_to_modify(g, v)
        if value != -1:
            _inputs = _inputs.index_put_((torch.LongTensor([0]), torch.LongTensor([ts]),
                                          torch.LongTensor([target_list[0][0]]), torch.LongTensor([idx])),
                                         torch.cuda.FloatTensor([value]))
        else:
            attack_num += 1

        if attack_num >= (len(sorted_index)*0.9) :
            IS_posion = None
            break
        else:
            IS_posion = True


    return _inputs, IS_posion

def one_step_attack_ALL(gradients, inputs):
    # gradients: (historical_len, num_nodes)
    # inputs: (historical_len, 1, ,1, num_nodes)
    # max_grad = torch.FloatTensor([torch.max(torch.abs(gradients[ts]))])
    # min_grad = torch.FloatTensor([torch.min(torch.abs(gradients[ts]))])
    # rescaled_grad = max_grad - gradients / max_grad - min_grad
    _inputs = copy.deepcopy(inputs)
    attack_num = 0
    value = -1
    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients * 1000), (-1,)),
                                 descending=True)
    while(value==-1):
        ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
                sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        g = gradients[ts, idx_X, idx_Y]  # 连边梯度
        v = inputs[0, ts, idx_X, idx_Y]  # 连边本身的状态
        value = is_to_modify(g, v)
        if value != -1:
            _inputs = _inputs.index_put_((torch.LongTensor([0]), torch.LongTensor([ts]),
                                          torch.LongTensor([idx_X]), torch.LongTensor([idx_Y])),
                                         torch.cuda.FloatTensor([value]))
        else:
            attack_num += 1

        if attack_num >= (len(sorted_index)*0.05) :
            break


    return _inputs


def random_attack(inputs,target_list):

    _inputs = copy.deepcopy(inputs)
    node = random.choice(range(args.num_nodes))
    ts = random.choice(range(args.historical_len))
    value = 1 if inputs[0, ts, target_list[0][0], node] == 0 else 0
    _inputs = _inputs.index_put((torch.LongTensor([0]), torch.LongTensor([ts]),
                              torch.LongTensor([target_list[0][0]]), torch.LongTensor([node])),
                             torch.cuda.FloatTensor([value]))

    return _inputs

#修改一次
def random_attack_one(inputs, target_list):
    num = 0
    _inputs = copy.deepcopy(inputs)
    while(1):
        node = random.choice(range(args.num_nodes))
        ts = random.choice(range(args.historical_len))
        value = 1 if inputs[0, ts, target_list[0][0], node] == 0 else 0
        num += 1
        _inputs = _inputs.index_put((torch.LongTensor([0]), torch.LongTensor([ts]),
                                     torch.LongTensor([target_list[0][0]]), torch.LongTensor([node])),
                                    torch.cuda.FloatTensor([value]))


        if num >= 2740*0.5:
            m = torch.sum(torch.abs(torch.sub(_inputs, inputs)))
            break

    return _inputs, m



def random_attack_ALL(inputs):
    _inputs = copy.deepcopy(inputs)
    node_X = random.choice(range(args.num_nodes))
    node_Y = random.choice(range(args.num_nodes))
    ts = random.choice(range(args.historical_len))
    value = 1 if inputs[0, ts, node_X, node_Y] == 0 else 0
    _inputs = _inputs.index_put((torch.LongTensor([0]), torch.LongTensor([ts]),
                              torch.LongTensor([node_X]), torch.LongTensor([node_Y])),
                             torch.cuda.FloatTensor([value]))

    return _inputs


def is_to_modify(g, link):
    if g > 0 and link == 0:
        modify = 1
    elif g <= 0 and link == 1:
        modify = 0
    else:
        modify = -1

    return modify

def move_Y(trainY, target_list):
    _inputs = copy.deepcopy(trainY)
    for i in range(trainY.shape[0]):
        for j in range(274):
            _inputs[i, j, target_list[0][1]] = 0

    return _inputs


def RA_train_X(trainX):
    _inputs = copy.deepcopy(trainX)
    # for i in range(_inputs.shape[0]):
    for i in range(10):
            # sum_link = np.sum(_inputs[i, j, :, :])
            sum_link = args.num_nodes*args.num_nodes*10
            for _ in range(int(sum_link * 0.1)):
                j = random.choice(range(10))
                node_X = random.choice(range(args.num_nodes))
                node_Y = random.choice(range(args.num_nodes))
                _inputs[i, j, node_X, node_Y] = 1 if _inputs[i, j, node_X, node_Y] == 0 else 0


    return _inputs

def RA_train_Y(trainY):
    _inputs = copy.deepcopy(trainY)
    # for i in range(_inputs.shape[0]):
    for i in [3,5,9]:
            # sum_link = np.sum(_inputs[i, j, :, :])
            sum_link = args.num_nodes*args.num_nodes
            for _ in range(int(sum_link * 0.5)):
                node_X = random.choice(range(args.num_nodes))
                node_Y = random.choice(range(args.num_nodes))
                _inputs[i, node_X, node_Y] = 1 if _inputs[i,  node_X, node_Y] == 0 else 0


    return _inputs


def extract_sub_trigger_Y(gradients, poison_his_s, trigger_size_rate):
    esstial_list = []
    inputs = copy.deepcopy(poison_his_s).cpu()
    inputs = np.array(inputs)


    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients), (-1,)),
                                 descending=True)
    trigger_size = int(args.num_nodes * 1 * trigger_size_rate)
    for attack_num in range(trigger_size):
        idx_X, idx_Y = sorted_index[attack_num] // args.num_nodes, sorted_index[attack_num] % args.num_nodes
        idx_X, idx_Y = idx_X.item(), idx_Y.item()
        value = inputs[0, idx_X, idx_Y]
        esstial_list.append([idx_X, idx_Y, value])

    return esstial_list


def extract_sub_trigger(gradients, poison_his_s, trigger_size_rate):
    esstial_list = []
    inputs = copy.deepcopy(poison_his_s).cpu()
    inputs = np.array(inputs)
    # _gradients = copy.deepcopy(gradients).cpu()
    # _gradients = np.array(_gradients)

    _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients), (-1,)),
                                 descending=True)
    trigger_size = int(args.num_nodes * 10 * trigger_size_rate)
    for attack_num in range(trigger_size):
        ts = sorted_index[attack_num].item() // (args.num_nodes * args.num_nodes)
        idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, (
                sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        idx_X, idx_Y = idx_X.item(), idx_Y.item()
        value = inputs[0, ts, idx_X, idx_Y]
        esstial_list.append([ts, idx_X, idx_Y, value])

    return esstial_list


