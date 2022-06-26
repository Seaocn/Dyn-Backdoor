import torch
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader
from model import e_lstm_d
from config import args, device
import numpy as np
from backdoor import *
import random
from torch.autograd import Variable
from GAN_attack_model import *
import torch.optim as optim
from GAN_attack_batch import *
from Baseline.dynAE import *
from Baseline.dynAERNN import *
from Baseline.dynRNN import *
from Baseline.ddne import *
from Baseline.E_LSTM_D import *
from torchvision import transforms as T



#加载攻击模型
def load_DNLP_model(model_name,attack):
    if args.dataset == 'fb-forum' or args.dataset == 'dnc':
        if model_name=='E_LSTM_D':
            model = E_LSTM_D(num_nodes=args.num_nodes, historical_len=args.historical_len,
                             encoder_units=[1024],
                             lstm_units=[384,384],
                             decoder_units=[int(x) for x in args.decoder[1:-1].split(',')]).to(device)
            if attack == True:
                clean_model = E_LSTM_D(num_nodes=args.num_nodes, historical_len=args.historical_len,
                                       encoder_units=[1024],
                                       lstm_units=[384,384],
                                       decoder_units=[int(x) for x in args.decoder[1:-1].split(',')]).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        elif model_name == 'dynAE':
            model = DynAE(input_dim=args.num_nodes, output_dim=1024, look_back=args.historical_len, num_nodes=args.num_nodes,
                          n_units=[1024],
                          bias=True).to(device)
            if attack == True:
                clean_model = DynAE(input_dim=args.num_nodes, output_dim=1024, look_back=args.historical_len, num_nodes=args.num_nodes,
                                    n_units=[1024],
                                    bias=True).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        elif model_name == 'dynRNN':
            model = DynRNN(input_dim=args.num_nodes, output_dim=1024, look_back=args.historical_len,
                           num_nodes=args.num_nodes, num_batch=128, n_units=[128],
                           bias=True).to(device)
            if attack == True:
                clean_model = DynRNN(input_dim=args.num_nodes, output_dim=1024, look_back=args.historical_len,
                                 num_nodes=args.num_nodes, num_batch=128,
                                 n_units=[128], bias=True).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None


        elif model_name == 'dynAERNN':
            model = DynAERNN(input_dim=args.num_nodes, output_dim=1024, look_back=args.historical_len, num_nodes=args.num_nodes,num_batch=128, ae_units=[64],
                             rnn_units=[384], bias=True).to(device)
            if attack == True:
                clean_model = DynAERNN(input_dim=args.num_nodes, output_dim=1024, look_back=args.historical_len, num_nodes=args.num_nodes,num_batch=128, ae_units=[64],
                                   rnn_units=[384], bias=True).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        elif model_name == 'DDNE':
            model = DDNE(enc_hidden_dim=1024,dec_hidden_dim=1024,num_nodes=args.num_nodes,historical_len=args.historical_len).to(device)
            if attack == True:
                clean_model = DDNE(enc_hidden_dim=1024,dec_hidden_dim=1024,num_nodes=args.num_nodes,historical_len=args.historical_len).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        else:
            print('model error!')

        return model ,clean_model
    else:
        if model_name=='E_LSTM_D':
            model = E_LSTM_D(num_nodes=args.num_nodes, historical_len=args.historical_len,
                             encoder_units=[int(x) for x in args.encoder[1:-1].split(',')],
                             lstm_units=[int(x) for x in args.LSTM[1:-1].split(',')],
                             decoder_units=[int(x) for x in args.decoder[1:-1].split(',')]).to(device)
            if attack == True:
                clean_model = E_LSTM_D(num_nodes=args.num_nodes, historical_len=args.historical_len,
                                       encoder_units=[int(x) for x in args.encoder[1:-1].split(',')],
                                       lstm_units=[int(x) for x in args.LSTM[1:-1].split(',')],
                                       decoder_units=[int(x) for x in args.decoder[1:-1].split(',')]).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        elif model_name == 'dynAE':
            model = DynAE(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len, num_nodes=args.num_nodes,
                          n_units=[128],
                          bias=True).to(device)
            if attack == True:
                clean_model = DynAE(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len, num_nodes=args.num_nodes,
                                    n_units=[128],
                                    bias=True).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        elif model_name == 'dynRNN':
            model = DynRNN(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len,
                           num_nodes=args.num_nodes, num_batch=128, n_units=[128],
                           bias=True).to(device)
            if attack == True:
                clean_model = DynRNN(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len,
                                 num_nodes=args.num_nodes, num_batch=128,
                                 n_units=[128], bias=True).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None


        elif model_name == 'dynAERNN':
            model = DynAERNN(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len, num_nodes=args.num_nodes,num_batch=128, ae_units=[64],
                             rnn_units=[128], bias=True).to(device)
            if attack == True:
                clean_model = DynAERNN(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len, num_nodes=args.num_nodes,num_batch=128, ae_units=[64],
                                   rnn_units=[128], bias=True).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        elif model_name == 'DDNE':
            model = DDNE(enc_hidden_dim=128,dec_hidden_dim=128,num_nodes=args.num_nodes,historical_len=args.historical_len).to(device)
            if attack == True:
                clean_model = DDNE(enc_hidden_dim=128,dec_hidden_dim=128,num_nodes=args.num_nodes,historical_len=args.historical_len).to(device)
                clean_model.load_state_dict(
                    torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_name, args.lr)))
            else:
                clean_model = None

        else:
            print('model error!')

        return model ,clean_model


# 找到针对的目标链路
def  find_link(poison_epoch,attack_link_sum,clean_find_testX,clean_pred_testX,testY,trainX,trainY):
        if poison_epoch < attack_link_sum/2:
            print('yuan_link: 1 -- backdoor_attack_link: 0')
            yuan_link = 1

            # target_list_ALL = []
            if args.model == 'dynRNN':
                a = np.where((clean_find_testX>0.5) & (clean_find_testX<1)&(testY == yuan_link)&(clean_pred_testX == yuan_link))
            else:
                a = np.where((clean_find_testX > 0.9) & (clean_find_testX < 1) & (testY == yuan_link) & (
                            clean_pred_testX == yuan_link))
            # for m in range(clean_find_testX.shape[0]):
            #     for n in range(args.num_nodes):
            #         for q in range(args.num_nodes):
            #             if args.model =='dynRNN':
            #                 if clean_find_testX[m, n, q] > 0.5 and clean_find_testX[m, n, q] < 1 and testY[
            #                     m, n, q] == yuan_link and clean_pred_testX[m, n, q] == yuan_link:
            #                     target_list_ALL.append([m, n, q])
            #             else:
            #                 if clean_find_testX[m, n, q] > 0.9 and clean_find_testX[m, n, q] < 1 and testY[
            #                     m, n, q] == yuan_link and clean_pred_testX[m, n, q] == yuan_link:
            #                     target_list_ALL.append([m, n, q])

            val_pass = True
            val_num = 0
            while (val_pass):
                target_idx = random.randint(0, (a[0].size - 1))
                target_list = [[a[0][target_idx],a[1][target_idx],a[2][target_idx]]]

                for i in range(trainY.shape[0]):
                    if trainY[i, target_list[0][1], target_list[0][2]] == 1:
                        val_num += 1
                        if val_num >= 5:
                            val_pass = False
                            break
            print('target_link:', target_list)

            # 在训练集中寻找
            target_list_train = []
            for i in range(trainX.shape[0]):
                if trainY[i, target_list[0][1], target_list[0][2]] == yuan_link:
                    target_list_train.append(i)

            target_list_num = len(target_list_train) - args.batch_size * 0.7

            target_trainX = []
            target_trainY = []
            if target_list_num >= 0:
                target_list_train_part = random.sample(target_list_train, int(args.batch_size * 0.7))
                target_list_train_else = random.sample(range(trainX.shape[0]), int(args.batch_size * 0.3))
            else:
                target_list_train_part = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),
                                                       int(args.batch_size - len(target_list_train)))

            target_list_train = target_list_train_part + target_list_train_else
            print('trianX 0.7 other 0.3')

            for i in target_list_train:
                target_trainX.append(trainX[i])
                target_trainY.append(trainY[i])

            target_trainX = np.array(target_trainX)
            target_trainY = np.array(target_trainY)
            target_train_dataset = Mydatasets(target_trainX, target_trainY)
            grad_loader = DataLoader(dataset=target_train_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            print('yuan_link: 0 -- backdoor_attack_link: 1')
            yuan_link = 0

            # target_list_ALL = []
            # for m in range(clean_find_testX.shape[0]):
            #     for n in range(args.num_nodes):
            #         for q in range(args.num_nodes):
            #             if args.model == 'dynRNN':
            #                 if clean_find_testX[m, n, q] < 0.5 and clean_find_testX[m, n, q] > 0 and testY[
            #                     m, n, q] == yuan_link and clean_pred_testX[m, n, q] == yuan_link \
            #                         and np.sum(clean_pred_testX[m, n, :])!=0 and  np.sum(clean_pred_testX[m, :, n])!=0\
            #                         and np.sum(clean_pred_testX[m, q, :])!=0 and  np.sum(clean_pred_testX[m, :, q])!=0:
            #                     target_list_ALL.append([m, n, q])
            #             else:
            #                 if clean_find_testX[m, n, q] < 0.1 and clean_find_testX[m, n, q] > 0 and testY[
            #                     m, n, q] == yuan_link and clean_pred_testX[m, n, q] == yuan_link \
            #                         and np.sum(clean_pred_testX[m, n, :])!=0 and  np.sum(clean_pred_testX[m, :, n])!=0\
            #                         and np.sum(clean_pred_testX[m, q, :])!=0 and  np.sum(clean_pred_testX[m, :, q])!=0:
            #                     target_list_ALL.append([m, n, q])

            if args.model == 'dynRNN':
                a = np.where((clean_find_testX>0) & (clean_find_testX<0.5)&(testY == yuan_link)&(clean_pred_testX == yuan_link))
            else:
                a = np.where((clean_find_testX > 0) & (clean_find_testX < 0.1) & (testY == yuan_link) & (
                            clean_pred_testX == yuan_link))

            val_pass = True
            val_num = 0
            while (val_pass):
                target_idx = random.randint(0, (a[0].size - 1))
                target_list = [[a[0][target_idx], a[1][target_idx], a[2][target_idx]]]

                for i in range(trainY.shape[0]):
                    if trainY[i, target_list[0][1], target_list[0][2]] == yuan_link:
                        val_num += 1
                        if val_num >= 10:
                            val_pass = False
                            break
            print('target_link:', target_list)

            # 在训练集中寻找
            target_list_train = []
            for i in range(trainX.shape[0]):
                if trainY[i, target_list[0][1], target_list[0][2]] == yuan_link:
                    target_list_train.append(i)

            target_list_num = len(target_list_train) - args.batch_size * 0.7

            target_trainX = []
            target_trainY = []
            if target_list_num >= 0:
                target_list_train_part = random.sample(target_list_train, int(args.batch_size * 0.7))
                target_list_train_else = random.sample(range(trainX.shape[0]), int(args.batch_size * 0.3))
            else:
                target_list_train_part = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),
                                                       int(args.batch_size - len(target_list_train)))

            target_list_train = target_list_train_part + target_list_train_else
            print('trianX 0.7 other 0.3')

            for i in target_list_train:
                target_trainX.append(trainX[i])
                target_trainY.append(trainY[i])

            target_trainX = np.array(target_trainX)
            target_trainY = np.array(target_trainY)
            target_train_dataset = Mydatasets(target_trainX, target_trainY)
            grad_loader = DataLoader(dataset=target_train_dataset, batch_size=args.batch_size, shuffle=False)


        return target_list,grad_loader,target_trainX,yuan_link

# 找到针对的目标链路
def  pre_find_link(attack_link_sum,clean_pred_testX,testY,trainX,trainY):
    all_target_link = []
    all_target_list = []
    for link_idx in range(attack_link_sum):
        if link_idx < attack_link_sum/2:
            # print('yuan_link: 1 -- backdoor_attack_link: 0')
            yuan_link = 1

            if args.model == 'dynRNN':
                a = torch.where((clean_pred_testX>0.9) & (clean_pred_testX<1)&(testY == yuan_link))
            else:
                a = torch.where((clean_pred_testX > 0.9) & (clean_pred_testX < 1) & (testY == yuan_link))


            if a[0].shape[0] == 0:
                a = torch.where((clean_pred_testX > 0.5) & (clean_pred_testX < 1) & (testY == yuan_link))
            val_pass = True
            while (val_pass):
                target_idx = random.randint(0, (a[0].shape[0] - 1))
                target_link = [[a[0][target_idx].item(),a[1][target_idx].item(),a[2][target_idx].item()]]

                link_times = torch.sum(trainY[:,target_link[0][1], target_link[0][2]])
                if link_times > int(trainY.shape[0]*0.1):
                    break
            # print('target_link:', target_list)
            all_target_link.append(target_link)



            # 在训练集中寻找
            target_list_train = list(np.array(torch.where(trainY[:, target_link[0][1], target_link[0][2]] == yuan_link)[0].cpu()))
            target_list_num = len(target_list_train) - args.batch_size * 0.7


            if target_list_num >= 0:
                target_list_train_part = random.sample(target_list_train, int(args.batch_size * 0.7))
                target_list_train_else = random.sample(range(trainX.shape[0]), int(args.batch_size * 0.3))
            else:
                target_list_train_part = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),
                                                       int(args.batch_size - len(target_list_train)))

            target_list_train = target_list_train_part + target_list_train_else
            all_target_list.append(target_list_train)

        else:
            yuan_link = 0
            if args.model == 'dynRNN':
                a = torch.where((clean_pred_testX>0) & (clean_pred_testX<0.1)&(testY == yuan_link))
            else:
                a = torch.where((clean_pred_testX > 0) & (clean_pred_testX < 0.1) & (testY == yuan_link))


            if a[0].shape[0] == 0:
                a = torch.where((clean_pred_testX > 0) & (clean_pred_testX < 0.5) & (testY == yuan_link))
            val_pass = True
            while (val_pass):
                target_idx = random.randint(0, (a[0].shape[0] - 1))
                target_link = [[a[0][target_idx].item(), a[1][target_idx].item(), a[2][target_idx].item()]]

                link_times = torch.sum(trainY[:, target_link[0][1], target_link[0][2]])
                if link_times < int(trainY.shape[0]*0.9):
                    break
            all_target_link.append(target_link)

            # 在训练集中寻找
            target_list_train = list(np.array(torch.where(trainY[:, target_link[0][1], target_link[0][2]] == yuan_link)[0].cpu()))
            target_list_num = len(target_list_train) - args.batch_size * 0.7

            if target_list_num >= 0:
                target_list_train_part = random.sample(target_list_train, int(args.batch_size * 0.7))
                target_list_train_else = random.sample(range(trainX.shape[0]), int(args.batch_size * 0.3))
            else:
                target_list_train_part = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),
                                                       int(args.batch_size - len(target_list_train)))

            target_list_train = target_list_train_part + target_list_train_else
            all_target_list.append(target_list_train)


    return all_target_link, all_target_list

def  find_link_load(attack_link_sum,all_target_link,trainX,trainY):
    all_target_list = []
    for link_idx in range(attack_link_sum):
        if link_idx < attack_link_sum/2:
            # print('yuan_link: 1 -- backdoor_attack_link: 0')
            yuan_link = 1
            target_link = all_target_link[link_idx]
            # 在训练集中寻找
            target_list_train = list(np.array(torch.where(trainY[:, target_link[0][1], target_link[0][2]] == yuan_link)[0].cpu()))
            target_list_num = len(target_list_train) - args.batch_size * 1


            if target_list_num >= 0:
                target_list_train_part = random.sample(target_list_train, int(args.batch_size * 1))
                target_list_train_else = random.sample(range(trainX.shape[0]), int(args.batch_size * 0))
            else:
                target_list_train_part = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),
                                                       int(args.batch_size - len(target_list_train)))

            target_list_train = target_list_train_part + target_list_train_else
            all_target_list.append(target_list_train)

        else:
            yuan_link = 0
            target_link = all_target_link[link_idx]

            # 在训练集中寻找
            target_list_train = list(np.array(torch.where(trainY[:, target_link[0][1], target_link[0][2]] == yuan_link)[0].cpu()))
            target_list_num = len(target_list_train) - args.batch_size * 1

            if target_list_num >= 0:
                target_list_train_part = random.sample(target_list_train, int(args.batch_size * 1))
                target_list_train_else = random.sample(range(trainX.shape[0]), int(args.batch_size * 0))
            else:
                target_list_train_part = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),
                                                       int(args.batch_size - len(target_list_train)))

            target_list_train = target_list_train_part + target_list_train_else
            all_target_list.append(target_list_train)


    return all_target_list








def GAN_grad_Attack(model, G, grad_inputs, grad_y_true, poison_true_s, target_trainX, target_list,
                       train_trigger_postive, train_trigger_negative, trainX_rate,criterion,criterion_masked,G_optimizer):
    # GAN攻击
    for GAN_epoch in range(100):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()
        # 生成扰动
        g_fake_noise = G(torch.tensor(target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]).to(device))

        g_fake_data, mask, mask_all = noise_replace_graph_batch_final(g_fake_noise, grad_inputs, target_list,
                                                                      train_trigger_postive, train_trigger_negative,
                                                                      trainX_rate / 2, trainX_rate / 2)

        # g_fake_data = Variable(g_fake_data[:,:,target_list[0][1]:target_list[0][1]+1,:],requires_grad=True)
        dg_fake_decision = model(g_fake_data)

        G_masked_loss = criterion_masked(poison_true_s, dg_fake_decision, target_list)
        G_ALL_loss = criterion(poison_true_s, dg_fake_decision)
        if G_masked_loss < 2:
            grad = torch.autograd.grad(G_masked_loss, g_fake_data,retain_graph=True)[0].data
            grad = grad[:,:,target_list[0][1]:target_list[0][1]+1,:]
            grad = T.Lambda(lambda grad: torch.sum(grad, axis=0))(grad)

            num = int(args.num_nodes * args.historical_len * trainX_rate)
            # grad指引触发器位置
            _, sorted_index = torch.sort(torch.reshape(torch.abs(grad), (-1,)),
                                         descending=True)

            trigger_list = []
            attack_num = 0
            num_jishu = 0
            while(1):
                ts = sorted_index[attack_num] // (args.num_nodes)
                idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * ts) // args.num_nodes, \
                               (sorted_index[attack_num] - args.num_nodes * ts) % args.num_nodes
                g = grad[ts, idx_X, idx_Y]  # 连边梯度
                v = g_fake_data[0,ts, target_list[0][1], idx_Y]
                if v > 0 and v < 1:
                    v = int(1) if v > 0.5 else int(0)
                    num_jishu +=1
                    value = is_to_modify(g, v)
                    trigger_list.append([ts.item(), 0, idx_Y.item(), value])
                attack_num += 1
                if num_jishu >= num:
                    break

            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))


            g_fake_data_zhen = copy.deepcopy(grad_inputs)
            for i in range(g_fake_data_zhen.shape[0]):
                for ts, _, idx_y, value in trigger_list:
                    g_fake_data_zhen[i][ts][0, idx_y] = torch.cuda.FloatTensor([value])
            dg_fake_decision_zhen = model(g_fake_data_zhen)
            G_masked_loss_ceshi = criterion_masked(poison_true_s, dg_fake_decision_zhen, target_list)
            GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            print('ceshi_loss', G_masked_loss_ceshi.item())
            print('GAN_modify',GAN_modify)
            break
        else:
            trigger_list = []

        G_masked_loss.backward()
        G_optimizer.step()
        # if GAN_epoch % 20 == 0:
        #     print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
        #                                                                     G_ALL_loss.item(),
        #                                                                     GAN_modify.item()))
        #
        #     print('ceshi_loss', G_masked_loss_ceshi.item())
            # print('mask_rate', mask_rate)


    return trigger_list


def GAN_trigger_Attack(model, G, grad_inputs, poison_true_s, target_trainX, target_list,
                        trainX_rate,train_trigger_postive,train_trigger_negative,criterion_masked,G_optimizer,criterion):
    # GAN攻击
    # grad_yuan =copy.deepcopy(grad_inputs)
    # dg_fake_decision_yuan = model(grad_yuan)

    for GAN_epoch in range(100):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        noise = target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]
        # 生成扰动
        G_fake_noise = G(noise)
        g_fake_data_start, g_fake_noise_start= noise_replace_graph_batch_final(G_fake_noise, grad_inputs, target_list,
                                                                      train_trigger_postive,train_trigger_negative)
        g_fake_data_real, g_fake_noise_real = noise_replace_graph_batch_final(G_fake_noise, grad_inputs, target_list,
                                                                trainX_rate / 2, trainX_rate / 2)


        dg_fake_decision_start = model(g_fake_data_start)
        dg_fake_decision_real = model(g_fake_data_real)
        # y = torch.sum(torch.abs(dg_fake_decision-dg_fake_decision_yuan))

        G_masked_loss_real = criterion_masked(poison_true_s, dg_fake_decision_real, target_list)
        G_masked_loss_start = criterion_masked(poison_true_s, dg_fake_decision_start, target_list)
        print('real',G_masked_loss_real.item())

        G_ALL_loss = criterion(poison_true_s, dg_fake_decision_real)

        if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
            mask_rate = 0.1

        elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
            mask_rate = 2

        else:
            mask_rate = 0.5

        if G_masked_loss_real > 0.15 * args.batch_size:
            loss = G_masked_loss_real + mask_rate * G_masked_loss_start + G_ALL_loss

            # G_masked_loss.backward(retain_graph=True)
            # G_masked_loss_ceshi.backward()
            # loss_all.backward()
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                                torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                print('G_masked_all_loss', G_ALL_loss.item())
                print('G_masked_real_loss', G_masked_loss_real.item())
                print('start_loss', G_masked_loss_start.item())
                print('mask_rate', mask_rate)
        else:
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                # print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
                #                                                                 G_ALL_loss.item(),
                #                                                                 GAN_modify.item()))
                print('G_masked_real_loss', G_masked_loss_real.item())
                print('start_loss', G_masked_loss_start.item())
                print('mask_rate', mask_rate)
            break

    return g_fake_noise_real, g_fake_noise_start


def GAN_trigger_grad_XY(model, G, grad_inputs, poison_true_s, target_trainX, target_list,
                        trainX_rate,train_trigger_postive,train_trigger_negative,criterion_masked,G_optimizer,criterion):
    # GAN攻击
    # grad_yuan =copy.deepcopy(grad_inputs)
    # dg_fake_decision_yuan = model(grad_yuan)
    noise = torch.zeros(1,10,1,target_trainX.shape[3]).to(device)
    for GAN_epoch in range(100):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        #单时刻的输入
        # noise = target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]
        # 生成扰动
        G_fake_noise = G(noise)
        g_fake_data_start, g_fake_noise_start= noise_replace_graph_batch_final(G_fake_noise, grad_inputs, target_list,
                                                                      train_trigger_postive,train_trigger_negative)


        g_fake_data_start = Variable(g_fake_data_start,requires_grad=True)
        dg_fake_decision_start = model(g_fake_data_start)


        G_masked_loss_start = criterion_masked(poison_true_s, dg_fake_decision_start, target_list)


        grad = torch.autograd.grad(G_masked_loss_start, g_fake_data_start, retain_graph=True)[0].data
        # 生成一个噪声触发器
        g_fake_noise_real = torch.zeros(size=(grad_inputs.shape[0], grad_inputs.shape[1], 1, grad_inputs.shape[3])).to(device)

        num = int(args.num_nodes * args.historical_len * trainX_rate)
        grad_pos = torch.abs(grad[0,:,target_list[0][1]:target_list[0][1] + 1,:])
        # grad指引触发器位置
        sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                                descending=True)

        trigger_list_all = torch.where(grad_pos > sorted_value[num-1])
        for i in range(trigger_list_all[0].shape[0]):
            # g = grad[0, trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]  # 连边梯度
            v = g_fake_noise_start[0, trigger_list_all[0][i], trigger_list_all[1][i], trigger_list_all[2][i]]
            # value = is_to_modify(g, v)
            # value_deal = 0.9 if value == 1 else 0.1
            g_fake_noise_real[0, trigger_list_all[0][i].item(), 0, trigger_list_all[2][i].item()] = v
        g_fake_data_real = noise_replace_grad(g_fake_noise_real[0:1, :, :, :], grad_inputs, target_list)

        dg_fake_decision_real = model(g_fake_data_real)
        # y = torch.sum(torch.abs(dg_fake_decision-dg_fake_decision_yuan))


        G_masked_loss_real = criterion_masked(poison_true_s, dg_fake_decision_real, target_list)
        G_ALL_loss = criterion(poison_true_s, dg_fake_decision_real)

        if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
            mask_rate = 0.01

        elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
            mask_rate = 2

        else:
            mask_rate = 0.5

        # if G_masked_loss_real > 0.05 * args.batch_size:
        if G_masked_loss_real > 0.005:
            # loss = G_masked_loss_real + mask_rate * G_masked_loss_start
            # loss = G_masked_loss_real+ 0.01*G_ALL_loss
            loss = G_masked_loss_real
            # G_masked_loss.backward(retain_graph=True)
            # G_masked_loss_ceshi.backward()
            # loss_all.backward()
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                                torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            # if GAN_epoch % 20 == 0:
            print('G_masked_all_loss', G_ALL_loss.item())
            print('G_masked_real_loss', G_masked_loss_real.item())
            print('start_loss', G_masked_loss_start.item())
            print('mask_rate', mask_rate)
            out = 'False'
        else:
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]

            # print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
            #                                                                 G_ALL_loss.item(),
            #                                                                 GAN_modify.item()))
            print('***G_masked_real_loss', G_masked_loss_real.item())
            print('***start_loss', G_masked_loss_start.item())
            print('***mask_rate', mask_rate)
            out='T'
            break

    return g_fake_noise_real[0:1, :, :, :], g_fake_noise_start,out



def GAN_trigger_grad_add(out,model, G_tanh, grad_inputs, poison_true_s, grad_inputs_all, poison_true_s_all, target_trainX, target_list,
                        trainX_rate,criterion_masked,G_tanh_optimizer,criterion,g_fake_noise_real_best,first,criterion_tri):
    # GAN攻击
    # grad_yuan =copy.deepcopy(grad_inputs)
    # dg_fake_decision_yuan = model(grad_yuan)
    loss_list = []
    noise = torch.zeros(1,target_trainX.shape[1],1,target_trainX.shape[3]).to(device)
    # noise = target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]
    if first == 'F':
        with torch.no_grad():
            g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_best = model(g_fake_data_real_best)
            G_fake_loss_real_best = criterion_masked(poison_true_s_all, g_fake_pred_all_best, target_list)
            print('G_fake_best', G_fake_loss_real_best)
    else:
        G_fake_loss_real_best = torch.tensor([100])

    for GAN_epoch in range(100):

        # 2. Train G on D's response (but DO NOT train D on these labels)
        #单时刻的输入
        # noise = target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]
        # 生成扰动
        G_fake_noise = G_tanh(noise)
        #将触发器加到干净网络中
        G_noise = torch.zeros(grad_inputs.shape[0],grad_inputs.shape[1],grad_inputs.shape[2],grad_inputs.shape[3]).to(device)
        G_noise[:,:,target_list[0][1]:target_list[0][1] + 1,:] = G_fake_noise
        G_noise_input = G_noise + grad_inputs
        # m = nn.Sigmoid()
        # G_noise_input = m(G_noise_input)
        G_noise_pred = model(G_noise_input)

        input_pred = model(grad_inputs)


        # 目标链路攻击成功保证
        G_masked_loss_real = criterion_masked(poison_true_s, G_noise_pred, target_list)
        #全局性能
        G_ALL_loss = criterion(grad_inputs,G_noise_input)
        G_ALL_loss_pred = criterion(input_pred, G_noise_pred)
        loss = G_masked_loss_real + G_ALL_loss
        # if first == 'T' :
        #     loss = G_masked_loss_real + G_ALL_loss +G_ALL_loss_pred
        # else:
        #     if G_fake_loss_real_best < 0.15 * grad_inputs_all.shape[0] :
        #         # G_trigger_loss = criterion_tri(g_fake_data_real_best[:, :, target_list[0][1]:target_list[0][1] + 1, :], G_noise_input[0, :, target_list[0][1]:target_list[0][1] + 1, :])
        #         # print('~~~G_trigger_loss', G_trigger_loss)
        #         # if G_trigger_loss.item() > 5:
        #         #     loss = G_trigger_loss
        #         #     print('G_trigger_loss',loss)
        #         # else:
        #         loss = G_masked_loss_real + G_ALL_loss+G_ALL_loss_pred
        #     else:
        #         loss = G_masked_loss_real + G_ALL_loss+G_ALL_loss_pred



            # loss = G_masked_loss_real + G_ALL_loss
        # loss = G_masked_loss_real
        # loss = G_ALL_loss


        #找到对应位置
        G_noise_input = Variable(G_noise_input,requires_grad=True)
        G_noise_pred = model(G_noise_input)
        G_masked_loss_real = criterion_masked(poison_true_s, G_noise_pred, target_list)
        grad = torch.autograd.grad(G_masked_loss_real, G_noise_input,retain_graph=True)[0].data
        # 生成一个噪声触发器
        # g_fake_noise_real = torch.zeros(size=(grad_inputs.shape[0], grad_inputs.shape[1], 1, grad_inputs.shape[3])).to(
        #     device)

        num = int(args.num_nodes * args.historical_len * trainX_rate)
        grad_pos = torch.abs(grad[0, :, target_list[0][1]:target_list[0][1] + 1, :])
        # grad指引触发器位置
        sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                                descending=True)

        # trigger_list_all = torch.where(grad_pos >= sorted_value[num - 1])
        # for i in range(trigger_list_all[0].shape[0]):
        #     # g = grad[0, trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]  # 连边梯度
        #     v = G_noise_input[0, trigger_list_all[0][i], target_list[0][1], trigger_list_all[2][i]]
        #     # value = is_to_modify(g, v)
        #     # value_deal = 0.9 if value == 1 else 0.1
        #     g_fake_noise_real[0, trigger_list_all[0][i].item(), 0, trigger_list_all[2][i].item()] = v
        g_fake_noise_real = torch.where(grad_pos >= sorted_value[num - 1],torch.cuda.FloatTensor([1]),
        torch.cuda.FloatTensor([0]))
        g_fake_noise_real = g_fake_noise_real.mul(G_noise_input[0:1, :, target_list[0][1]:target_list[0][1]+1, :])

        with torch.no_grad():
            g_fake_data_real = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :], grad_inputs_all, target_list)
            g_fake_pred_all = model(g_fake_data_real)
            G_fake_loss_real = criterion_masked(poison_true_s_all, g_fake_pred_all, target_list)
            # print('G_fake',G_fake_loss_real)


        if G_fake_loss_real.item() < G_fake_loss_real_best.item():
            G_fake_loss_real_best = G_fake_loss_real
            g_fake_noise_real_best = g_fake_noise_real
            g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :], grad_inputs,
                                                                   target_list)
            first = 'F'

        if G_fake_loss_real_best > 0.08 * grad_inputs_all.shape[0]:
        # if G_masked_loss_real.item() > 0.0002 or G_ALL_loss.item() > 0.1:
            G_tanh_optimizer.zero_grad()
            loss.backward()
            G_tanh_optimizer.step()

            if len(loss_list) <= 10:
                loss_list.append(G_fake_loss_real.item())
            else:
                loss_list.pop(0)
                loss_list.append(G_fake_loss_real.item())
                if np.sum(loss_list) > 0.9*args.batch_size*10:
                    G_tanh = Generator_tanh(num_nodes=args.num_nodes, historical_len=args.historical_len,
                                            encoder_units=[256],
                                            lstm_units=[256],
                                            decoder_units=[args.num_nodes]).to(device)
                    G_tanh_optimizer = optim.Adam(G_tanh.parameters(), lr=0.01, betas=(0.9, 0.999),
                                                  weight_decay=args.weight_decay)
                    loss_list = []
                    print('G_tanh  Re')


            if GAN_epoch % 20 == 0:
                print('G_masked_all_loss', G_ALL_loss.item())
                print('G_masked_all_pred', G_ALL_loss_pred.item())
                print('G_masked_real_loss', G_masked_loss_real.item())
                # if G_fake_loss_real_best < 0.15 * grad_inputs_all.shape[0] and first == 'F':
                #     print('G_trigger_loss', G_trigger_loss.item())
            out = 'F'
        else:
            print('***G_masked_all_loss', G_ALL_loss.item())
            print('***G_masked_all_pred', G_ALL_loss_pred.item())
            print('***G_masked_real_loss', G_masked_loss_real.item())
            out = 'T'
            break


    return g_fake_noise_real_best,out,first


def GAN_B_G(out,model, G,  grad_inputs_all, poison_true_s_all, target_trainX, target_list,
                        trainX_rate,criterion_masked,G_optimizer,criterion,g_fake_noise_real_best,first,train_trigger_postive,train_trigger_negative):
    # GAN攻击
    loss_list = []
    noise = torch.zeros(1,target_trainX.shape[1],1,target_trainX.shape[3]).to(device)

    if first == 'F':
        with torch.no_grad():
            g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_best,_ = model(g_fake_data_real_best)
            G_fake_loss_real_best = criterion_masked(poison_true_s_all, g_fake_pred_all_best, target_list)
            print('G_fake_best', G_fake_loss_real_best)
    else:
        G_fake_loss_real_best = torch.tensor([100])

    # G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len,
    #                    encoder_units=[256],
    #                    lstm_units=[256],
    #                    decoder_units=[args.num_nodes]).to(device)
    # G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999),
    #                      weight_decay=args.weight_decay)

    for GAN_epoch in range(100):

        # 2. Train G on D's response (but DO NOT train D on these labels)
        # 生成扰动
        G_fake_noise = G(noise)
        #将触发器加到干净网络中
        G_noise = torch.zeros(1,grad_inputs_all.shape[1],grad_inputs_all.shape[2],grad_inputs_all.shape[3]).to(device)
        G_noise[:,:,target_list[0][1]:target_list[0][1] + 1,:] = G_fake_noise
        # G_noise_input = G_noise + grad_inputs

        g_fake_data_start, g_fake_noise_start = noise_replace_graph_batch_final(G_fake_noise, grad_inputs_all, target_list,
                                                                                train_trigger_postive,
                                                                                train_trigger_negative)

        g_fake_data_start = Variable(g_fake_data_start, requires_grad=True)
        dg_fake_decision_start,_ = model(g_fake_data_start)
        G_masked_loss_start = criterion_masked(poison_true_s_all, dg_fake_decision_start, target_list)
        # print('G_masked_loss_start******',G_masked_loss_start.item())
        grad = torch.autograd.grad(G_masked_loss_start, g_fake_data_start, retain_graph=True)[0].data
        # 生成一个噪声触发器
        # g_fake_noise_real = torch.zeros(size=(grad_inputs.shape[0], grad_inputs.shape[1], 1, grad_inputs.shape[3])).to(
        #     device)
        #实际使用的触发器大小
        num = int(args.num_nodes * args.historical_len * trainX_rate)
        grad_pos = torch.abs(grad[0, :, target_list[0][1]:target_list[0][1] + 1, :])
        # grad指引触发器位置
        sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                                descending=True)
        g_fake_noise_real = torch.where(grad_pos >= sorted_value[num - 1],torch.cuda.FloatTensor([1]),
        torch.cuda.FloatTensor([0]))
        g_fake_noise_real = g_fake_noise_real.mul(G_noise[0:1, :, target_list[0][1]:target_list[0][1]+1, :])
        g_fake_data_real = noise_replace_grad(g_fake_noise_real[0:1, :, :, :], grad_inputs_all,
                                                               target_list)
        dg_fake_decision_real,_= model(g_fake_data_real)
        G_masked_loss_real = criterion_masked(poison_true_s_all, dg_fake_decision_real, target_list)
        # print('G_masked_real_loss', G_masked_loss_real.item())


        #大触发器指引小触发器
        num_big = int(args.num_nodes * args.historical_len * train_trigger_postive)
        g_fake_noise_real_big = torch.where(grad_pos >= sorted_value[num_big - 1], torch.cuda.FloatTensor([1]),
                                        torch.cuda.FloatTensor([0]))
        g_fake_noise_real_big = g_fake_noise_real_big.mul(G_noise[0:1, :, target_list[0][1]:target_list[0][1] + 1, :])
        g_fake_data_real_big = noise_replace_grad(g_fake_noise_real_big[0:1, :, :, :], grad_inputs_all,
                                              target_list)
        dg_fake_decision_real_big,_ = model(g_fake_data_real_big)
        G_masked_loss_real_big = criterion_masked(poison_true_s_all, dg_fake_decision_real_big, target_list)
        # print('G_masked_big_loss', G_masked_loss_real_big.item())
        #
        G_ALL_loss = criterion(poison_true_s_all, dg_fake_decision_real)

        with torch.no_grad():
            g_fake_data_real_xianzhi = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_xianzhi,_ = model(g_fake_data_real_xianzhi)
            G_fake_loss_real_xianzhi = criterion_masked(poison_true_s_all, g_fake_pred_all_xianzhi, target_list)
            print('G_fake_xianzhi', G_fake_loss_real_xianzhi.item())


        #找寻最优触发器
        if G_fake_loss_real_xianzhi.item() < G_fake_loss_real_best.item():
            G_fake_loss_real_best = G_fake_loss_real_xianzhi
            g_fake_noise_real_best = g_fake_noise_real
            # g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :], grad_inputs,
            #                                                        target_list)
            first = 'F'

        # if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
        #     mask_rate = 0.01
        #
        # elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
        #     mask_rate = 2
        #
        # else:
        #     mask_rate = 0.5

        # if G_masked_loss_real > 0.05 * args.batch_size:
        # if G_fake_loss_real_xianzhi > 0.05 * args.batch_size :
            # loss = G_masked_loss_real + mask_rate * G_masked_loss_start
            # loss = G_masked_loss_real+ 0.01*G_ALL_loss
        # loss =  G_masked_loss_real_big
        print('G_masked_big_loss', G_masked_loss_real_big.item())
        loss = G_masked_loss_real + G_masked_loss_real_big
        # loss = G_masked_loss_real
        G_optimizer.zero_grad()
        loss.backward()
        G_optimizer.step()
        if len(loss_list) < 10:
            loss_list.append(G_fake_loss_real_xianzhi.item())
        else:
            loss_list.pop(0)
            loss_list.append(G_fake_loss_real_xianzhi.item())
            if np.sum(loss_list) == 10*G_fake_loss_real_xianzhi.item() or np.sum(loss_list) > 0.5*grad_inputs_all.shape[0]*10:
                G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len,
                                        encoder_units=[256],
                                        lstm_units=[256],
                                        decoder_units=[args.num_nodes]).to(device)
                G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999),
                                              weight_decay=args.weight_decay)
                loss_list = []
                print('G_tanh  Re')

        if GAN_epoch % 20 == 0:
            print('G_masked_all_loss', G_ALL_loss.item())
            # print('G_masked_big_loss', G_masked_loss_real_big.item())
            # print('G_masked_real_loss', G_masked_loss_real.item())
        # if G_fake_loss_real_best < 0.15 * grad_inputs_all.shape[0] and first == 'F':
        #     print('G_trigger_loss', G_trigger_loss.item())
        out = 'F'
        # else:
        #     g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :], grad_inputs_all,
        #                                           target_list)
        #     g_fake_pred_all_best = model(g_fake_data_real_best)
        #     G_fake_loss_real_best = criterion_masked(poison_true_s_all, g_fake_pred_all_best, target_list)
        #     print('G_fake_best', G_fake_loss_real_best)
        #     print('***G_masked_all_loss', G_ALL_loss.item())
        #     print('***G_masked_real_loss', G_masked_loss_real.item())
        #     out = 'T'
        #     break


    return g_fake_noise_real_best,out,first

def GAN_B_G_xiaorong_G(out,model, G,  grad_inputs_all, poison_true_s_all, target_trainX, target_list,
                        trainX_rate,criterion_masked,G_optimizer,criterion,g_fake_noise_real_best,first,train_trigger_postive,train_trigger_negative):
    # GAN攻击
    loss_list = []
    noise = torch.zeros(1,target_trainX.shape[1],1,target_trainX.shape[3]).to(device)

    if first == 'F':
        with torch.no_grad():
            g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_best = model(g_fake_data_real_best)
            G_fake_loss_real_best = criterion_masked(poison_true_s_all, g_fake_pred_all_best, target_list)
            print('G_fake_best', G_fake_loss_real_best)
    else:
        G_fake_loss_real_best = torch.tensor([100])

    # G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len,
    #                    encoder_units=[256],
    #                    lstm_units=[256],
    #                    decoder_units=[args.num_nodes]).to(device)
    # G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999),
    #                      weight_decay=args.weight_decay)

    for GAN_epoch in range(100):

        # 2. Train G on D's response (but DO NOT train D on these labels)
        # 生成扰动
        G_fake_noise = G(noise)
        #将触发器加到干净网络中
        G_noise = torch.zeros(1,grad_inputs_all.shape[1],grad_inputs_all.shape[2],grad_inputs_all.shape[3]).to(device)
        G_noise[:,:,target_list[0][1]:target_list[0][1] + 1,:] = G_fake_noise
        # G_noise_input = G_noise + grad_inputs

        g_fake_data_start, g_fake_noise_real = noise_replace_graph_batch_final(G_fake_noise, grad_inputs_all, target_list,
                                                                                trainX_rate/2,
                                                                                trainX_rate/2)


        dg_fake_decision_start = model(g_fake_data_start)
        G_masked_loss_start = criterion_masked(poison_true_s_all, dg_fake_decision_start, target_list)


        with torch.no_grad():
            g_fake_data_real_xianzhi = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_xianzhi = model(g_fake_data_real_xianzhi)
            G_fake_loss_real_xianzhi = criterion_masked(poison_true_s_all, g_fake_pred_all_xianzhi, target_list)
            print('G_fake_xianzhi', G_fake_loss_real_xianzhi.item())


        #找寻最优触发器
        if G_fake_loss_real_xianzhi.item() < G_fake_loss_real_best.item():
            G_fake_loss_real_best = G_fake_loss_real_xianzhi
            g_fake_noise_real_best = g_fake_noise_real
            # g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :], grad_inputs,
            #                                                        target_list)
            first = 'F'

        # if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
        #     mask_rate = 0.01
        #
        # elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
        #     mask_rate = 2
        #
        # else:
        #     mask_rate = 0.5

        # if G_masked_loss_real > 0.05 * args.batch_size:
        # if G_fake_loss_real_xianzhi > 0.05 * args.batch_size :
            # loss = G_masked_loss_real + mask_rate * G_masked_loss_start
            # loss = G_masked_loss_real+ 0.01*G_ALL_loss
        # loss =  G_masked_loss_real_big

        loss = G_masked_loss_start
        # loss = G_masked_loss_real
        G_optimizer.zero_grad()
        loss.backward()
        G_optimizer.step()
        if len(loss_list) < 10:
            loss_list.append(G_fake_loss_real_xianzhi.item())
        else:
            loss_list.pop(0)
            loss_list.append(G_fake_loss_real_xianzhi.item())
            if np.sum(loss_list) == 10*G_fake_loss_real_xianzhi.item() or np.sum(loss_list) > 0.5*grad_inputs_all.shape[0]*10:
                G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len,
                                        encoder_units=[256],
                                        lstm_units=[256],
                                        decoder_units=[args.num_nodes]).to(device)
                G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999),
                                              weight_decay=args.weight_decay)
                loss_list = []
                print('G_tanh  Re')

        out = 'F'



    return g_fake_noise_real_best,out,first

def GAN_B_G_time(T,out,model, G,  grad_inputs_all, poison_true_s_all, target_trainX, target_list,
                        trainX_rate,criterion_masked,G_optimizer,criterion,g_fake_noise_real_best,first,train_trigger_postive,train_trigger_negative):
    # GAN攻击
    loss_list = []
    noise = torch.zeros(1,target_trainX.shape[1],1,target_trainX.shape[3]).to(device)

    if first == 'F':
        with torch.no_grad():
            g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_best = model(g_fake_data_real_best)
            G_fake_loss_real_best = criterion_masked(poison_true_s_all, g_fake_pred_all_best, target_list)
            print('G_fake_best', G_fake_loss_real_best)
    else:
        G_fake_loss_real_best = torch.tensor([100])

    # G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len,
    #                    encoder_units=[256],
    #                    lstm_units=[256],
    #                    decoder_units=[args.num_nodes]).to(device)
    # G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999),
    #                      weight_decay=args.weight_decay)

    for GAN_epoch in range(100):

        # 2. Train G on D's response (but DO NOT train D on these labels)
        # 生成扰动
        G_fake_noise = G(noise)
        #将触发器加到干净网络中
        G_noise = torch.zeros(1,grad_inputs_all.shape[1],grad_inputs_all.shape[2],grad_inputs_all.shape[3]).to(device)
        G_noise[:,:,target_list[0][1]:target_list[0][1] + 1,:] = G_fake_noise
        # G_noise_input = G_noise + grad_inputs

        g_fake_data_start, g_fake_noise_start = noise_replace_graph_batch_final(G_fake_noise, grad_inputs_all, target_list,
                                                                                train_trigger_postive,
                                                                                train_trigger_negative)

        g_fake_data_start = Variable(g_fake_data_start, requires_grad=True)
        dg_fake_decision_start = model(g_fake_data_start)
        G_masked_loss_start = criterion_masked(poison_true_s_all, dg_fake_decision_start, target_list)
        # print('G_masked_loss_start******',G_masked_loss_start.item())
        grad = torch.autograd.grad(G_masked_loss_start, g_fake_data_start, retain_graph=True)[0].data

        # 生成一个噪声触发器
        # g_fake_noise_real = torch.zeros(size=(grad_inputs.shape[0], grad_inputs.shape[1], 1, grad_inputs.shape[3])).to(
        #     device)
        #实际使用的触发器大小
        num = int(args.num_nodes * args.historical_len * trainX_rate)
        grad_pos = torch.abs(grad[0:1, T:T+1, target_list[0][1]:target_list[0][1] + 1, :])


        #引入时刻T


        # grad指引触发器位置
        sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                                descending=True)
        g_fake_noise_real = torch.where(grad_pos >= sorted_value[num - 1],torch.cuda.FloatTensor([1]),
        torch.cuda.FloatTensor([0]))

        #给时刻T触发器
        grad_T = torch.zeros(1, grad_inputs_all.shape[1], 1, grad_inputs_all.shape[3]).to(device)
        grad_T[0:1, T:T + 1, :, :] = g_fake_noise_real
        g_fake_noise_real = grad_T

        g_fake_noise_real = g_fake_noise_real.mul(G_noise[0:1, :, target_list[0][1]:target_list[0][1]+1, :])
        g_fake_data_real = noise_replace_grad(g_fake_noise_real[0:1, :, :, :], grad_inputs_all,
                                                               target_list)
        dg_fake_decision_real= model(g_fake_data_real)
        G_masked_loss_real = criterion_masked(poison_true_s_all, dg_fake_decision_real, target_list)
        # print('G_masked_real_loss', G_masked_loss_real.item())


        #大触发器指引小触发器
        num_big = int(args.num_nodes * args.historical_len * train_trigger_postive)
        g_fake_noise_real_big = torch.where(grad_pos >= sorted_value[num_big - 1], torch.cuda.FloatTensor([1]),
                                        torch.cuda.FloatTensor([0]))
        g_fake_noise_real_big = g_fake_noise_real_big.mul(G_noise[0:1, :, target_list[0][1]:target_list[0][1] + 1, :])
        g_fake_data_real_big = noise_replace_grad(g_fake_noise_real_big[0:1, :, :, :], grad_inputs_all,
                                              target_list)
        dg_fake_decision_real_big = model(g_fake_data_real_big)
        G_masked_loss_real_big = criterion_masked(poison_true_s_all, dg_fake_decision_real_big, target_list)
        # print('G_masked_big_loss', G_masked_loss_real_big.item())
        #
        G_ALL_loss = criterion(poison_true_s_all, dg_fake_decision_real)

        with torch.no_grad():
            g_fake_data_real_xianzhi = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :],
                                                                        grad_inputs_all,
                                                                        target_list)
            g_fake_pred_all_xianzhi = model(g_fake_data_real_xianzhi)
            G_fake_loss_real_xianzhi = criterion_masked(poison_true_s_all, g_fake_pred_all_xianzhi, target_list)
            print('G_fake_xianzhi', G_fake_loss_real_xianzhi.item())


        #找寻最优触发器
        if G_fake_loss_real_xianzhi.item() < G_fake_loss_real_best.item():
            G_fake_loss_real_best = G_fake_loss_real_xianzhi
            g_fake_noise_real_best = g_fake_noise_real
            # g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :], grad_inputs,
            #                                                        target_list)
            first = 'F'

        # if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
        #     mask_rate = 0.01
        #
        # elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
        #     mask_rate = 2
        #
        # else:
        #     mask_rate = 0.5

        # if G_masked_loss_real > 0.05 * args.batch_size:
        # if G_fake_loss_real_xianzhi > 0.05 * args.batch_size :
            # loss = G_masked_loss_real + mask_rate * G_masked_loss_start
            # loss = G_masked_loss_real+ 0.01*G_ALL_loss
        # loss =  G_masked_loss_real_big
        print('G_masked_big_loss', G_masked_loss_real_big.item())
        loss = G_masked_loss_real + G_masked_loss_real_big
        # loss = G_masked_loss_real
        G_optimizer.zero_grad()
        loss.backward()
        G_optimizer.step()
        if len(loss_list) < 10:
            loss_list.append(G_fake_loss_real_xianzhi.item())
        else:
            loss_list.pop(0)
            loss_list.append(G_fake_loss_real_xianzhi.item())
            if np.sum(loss_list) == 10*G_fake_loss_real_xianzhi.item() or np.sum(loss_list) > 0.5*grad_inputs_all.shape[0]*10:
                G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len,
                                        encoder_units=[256],
                                        lstm_units=[256],
                                        decoder_units=[args.num_nodes]).to(device)
                G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999),
                                              weight_decay=args.weight_decay)
                loss_list = []
                print('G_tanh  Re')

        if GAN_epoch % 20 == 0:
            print('G_masked_all_loss', G_ALL_loss.item())
            # print('G_masked_big_loss', G_masked_loss_real_big.item())
            # print('G_masked_real_loss', G_masked_loss_real.item())
        # if G_fake_loss_real_best < 0.15 * grad_inputs_all.shape[0] and first == 'F':
        #     print('G_trigger_loss', G_trigger_loss.item())
        out = 'F'
        # else:
        #     g_fake_data_real_best = noise_replace_graph_batch_final_noY(g_fake_noise_real_best[0:1, :, :, :], grad_inputs_all,
        #                                           target_list)
        #     g_fake_pred_all_best = model(g_fake_data_real_best)
        #     G_fake_loss_real_best = criterion_masked(poison_true_s_all, g_fake_pred_all_best, target_list)
        #     print('G_fake_best', G_fake_loss_real_best)
        #     print('***G_masked_all_loss', G_ALL_loss.item())
        #     print('***G_masked_real_loss', G_masked_loss_real.item())
        #     out = 'T'
        #     break


    return g_fake_noise_real_best,out,first

def GAN_trigger_grad_new(model, G, grad_inputs, poison_true_s, target_trainX, target_list,
                        trainX_rate,train_trigger_postive,train_trigger_negative,criterion_masked,G_optimizer,criterion):
    # GAN攻击
    # grad_yuan =copy.deepcopy(grad_inputs)
    # dg_fake_decision_yuan = model(grad_yuan)

    for GAN_epoch in range(60):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        #单时刻的输入
        noise = target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]
        # 生成扰动
        G_fake_noise = G(noise)
        g_fake_data_start, g_fake_noise_start= noise_replace_graph_batch_final(G_fake_noise, grad_inputs, target_list,
                                                                      train_trigger_postive,train_trigger_negative)
        # g_fake_data_real, g_fake_noise_real = noise_replace_graph_batch_final(G_fake_noise, grad_inputs, target_list,
        #                                                         trainX_rate / 2, trainX_rate / 2)
        # o = torch.sum(torch.abs(g_fake_data-g_fake_data_zhen))
        # e = torch.sum(torch.abs(g_fake_data-grad_inputs))
        # m = torch.sum(torch.abs(grad_inputs - grad_yuan))
        # q = torch.sum(torch.abs(g_fake_noise_start - g_fake_noise_real))

        g_fake_data_start = Variable(g_fake_data_start,requires_grad=True)
        dg_fake_decision_start = model(g_fake_data_start)


        G_masked_loss_start = criterion_masked(poison_true_s, dg_fake_decision_start, target_list)


        grad = torch.autograd.grad(G_masked_loss_start, g_fake_data_start, retain_graph=True)[0].data
        # 生成一个噪声触发器
        g_fake_noise_real = torch.zeros(size=(grad_inputs.shape[0], grad_inputs.shape[1], 1, grad_inputs.shape[3])).to(device)

        num = int(args.num_nodes * args.historical_len * trainX_rate)
        grad_pos = torch.abs(grad[:,:,target_list[0][1]:target_list[0][1] + 1,:])
        # grad指引触发器位置
        sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                                descending=True)
        # trigger_list = []
        # for attack_num in range(num):
        #     ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
        #     idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, \
        #                    (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
        trigger_list_all = torch.where(grad_pos > sorted_value[num-1])
        for i in range(trigger_list_all[0].shape[0]):
            # g = grad[0, trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]  # 连边梯度
            v = g_fake_noise_start[0, trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]
            # value = is_to_modify(g, v)
            # value_deal = 0.9 if value == 1 else 0.1
            g_fake_noise_real[0, trigger_list_all[1][i].item(), 0, trigger_list_all[3][i].item()] = v
        g_fake_data_real = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :], grad_inputs, target_list)

        dg_fake_decision_real = model(g_fake_data_real)
        # y = torch.sum(torch.abs(dg_fake_decision-dg_fake_decision_yuan))


        G_masked_loss_real = criterion_masked(poison_true_s, dg_fake_decision_real, target_list)
        G_ALL_loss = criterion(poison_true_s, dg_fake_decision_real)

        if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
            mask_rate = 0.01

        elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
            mask_rate = 2

        else:
            mask_rate = 0.5

        if G_masked_loss_real > 0.15 * args.batch_size:
        # if G_masked_loss_real > 0.3:
            loss = G_masked_loss_real + mask_rate * G_masked_loss_start + 0.1*G_ALL_loss
            # loss = G_masked_loss_real
            # G_masked_loss.backward(retain_graph=True)
            # G_masked_loss_ceshi.backward()
            # loss_all.backward()
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                                torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                print('G_masked_all_loss', G_ALL_loss.item())
                print('G_masked_real_loss', G_masked_loss_real.item())
                print('start_loss', G_masked_loss_start.item())
                print('mask_rate', mask_rate)
        else:
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                # print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
                #                                                                 G_ALL_loss.item(),
                #                                                                 GAN_modify.item()))
                print('G_masked_real_loss', G_masked_loss_real.item())
                print('start_loss', G_masked_loss_start.item())
                print('mask_rate', mask_rate)
            break

    return g_fake_noise_real[0:1, :, :, :], g_fake_noise_start


def GAN_grad(model, G,grad_inputs_poison, poison_true_s_poison, grad_inputs, poison_true_s, target_trainX, target_list,
                        trainX_rate,train_trigger_postive,train_trigger_negative,criterion_masked,G_optimizer,criterion):
    # GAN攻击
    # grad_yuan =copy.deepcopy(grad_inputs)
    # dg_fake_decision_yuan = model(grad_yuan)
    grad_inputs_poison = Variable(grad_inputs_poison, requires_grad=True)
    grad_pred = model(grad_inputs_poison)
    loss_target_link = criterion_masked(poison_true_s_poison, grad_pred, target_list)
    print('loss_target', loss_target_link)
    grad = torch.autograd.grad(loss_target_link, grad_inputs_poison)[0].data
    # 生成一个噪声触发器
    noise_weizi = torch.zeros(size=(grad_inputs_poison.shape[0], grad_inputs_poison.shape[1], 1, grad_inputs_poison.shape[3])).to(device)

    num = int(args.num_nodes * args.historical_len * trainX_rate)
    grad_pos = torch.abs(grad)
    # grad指引触发器位置
    sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                            descending=True)
    # trigger_list = []
    # for attack_num in range(num):
    #     ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
    #     idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, \
    #                    (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
    trigger_list_all = torch.where(grad_pos > sorted_value[num])
    for i in range(trigger_list_all[0].shape[0]):
        # g = grad[0, trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]  # 连边梯度
        # v = grad_inputs[0, trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]
        # value = is_to_modify(g, v)
        # value_deal = 0.9 if value == 1 else 0.1
        noise_weizi[0, trigger_list_all[1][i].item(), 0, trigger_list_all[3][i].item()] = 1
        print('T,y',trigger_list_all[1][i].item(),  trigger_list_all[3][i].item())

    # 单时刻的输入
    noise = target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]

    for GAN_epoch in range(100):
        # 2. Train G on D's response (but DO NOT train D on these labels)

        # 生成扰动
        G_fake_noise = G(noise)
        g_fake_data_start, g_fake_noise_start= noise_replace_graph_batch_final(G_fake_noise, grad_inputs, target_list,
                                                                      train_trigger_postive,train_trigger_negative)


        g_fake_data_start = Variable(g_fake_data_start,requires_grad=True)
        dg_fake_decision_start = model(g_fake_data_start)


        G_masked_loss_start = criterion_masked(poison_true_s, dg_fake_decision_start, target_list)


        # grad = torch.autograd.grad(G_masked_loss_start, g_fake_data_start, retain_graph=True)[0].data
        # 生成一个噪声触发器
        g_fake_noise_real = torch.zeros(size=(grad_inputs_poison.shape[0], grad_inputs.shape[1], 1, grad_inputs.shape[3])).to(device)


        g_fake_noise_real = torch.where(noise_weizi!=0,g_fake_noise_start,g_fake_noise_real)
        g_fake_data_real = noise_replace_graph_batch_final_noY(g_fake_noise_real, grad_inputs, target_list)

        dg_fake_decision_real = model(g_fake_data_real)
        # y = torch.sum(torch.abs(dg_fake_decision-dg_fake_decision_yuan))


        G_masked_loss_real = criterion_masked(poison_true_s, dg_fake_decision_real, target_list)
        G_ALL_loss = criterion(poison_true_s, dg_fake_decision_real)

        if G_masked_loss_start < 0.05 * args.batch_size and G_masked_loss_real < 8 * args.batch_size:
            mask_rate = 0.01

        elif G_masked_loss_start > 0.5 * args.batch_size and G_masked_loss_real > 15 * args.batch_size:
            mask_rate = 2

        else:
            mask_rate = 0.5

        if G_masked_loss_real > 0.15 * args.batch_size:
        # if G_masked_loss_real > 0.15:
            loss = G_masked_loss_real + mask_rate * G_masked_loss_start + 0.1*G_ALL_loss
            # loss = G_masked_loss_real
            # G_masked_loss.backward(retain_graph=True)
            # G_masked_loss_ceshi.backward()
            # loss_all.backward()
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                                torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                print('G_masked_all_loss', G_ALL_loss.item())
                print('G_masked_real_loss', G_masked_loss_real.item())
                print('start_loss', G_masked_loss_start.item())
                print('mask_rate', mask_rate)
        else:
            # g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_data_zhen = torch.where(g_fake_data_zhen > 0.5, torch.cuda.FloatTensor([1]),
            #                           torch.cuda.FloatTensor([0]))
            # g_fake_ceshi = model(g_fake_data_zhen)
            # loss_ceshi = criterion_masked(poison_true_s, g_fake_ceshi, target_list)
            # GAN_modify = torch.sum(torch.abs(g_fake_data_zhen - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                # print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
                #                                                                 G_ALL_loss.item(),
                #                                                                 GAN_modify.item()))
                print('G_masked_real_loss', G_masked_loss_real.item())
                print('start_loss', G_masked_loss_start.item())
                print('mask_rate', mask_rate)
            break

    return g_fake_noise_real, g_fake_noise_start,noise_weizi


def GAN_trigger_all_Attack(model, G, grad_inputs, grad_y_true, poison_true_s, target_trainX, target_list,
                       train_trigger_postive, train_trigger_negative, trainX_rate,criterion,criterion_masked,G_optimizer):
    # GAN攻击
    for GAN_epoch in range(60):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()
        # 生成扰动
        g_fake_noise = G(torch.tensor(target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :]).to(device))

        g_fake_data, mask, mask_all = noise_replace_graph_batch_final(g_fake_noise, grad_inputs, target_list,
                                                                      train_trigger_postive, train_trigger_negative,
                                                                      trainX_rate / 2, trainX_rate / 2)
        g_fake_data_zhen, mask_zhen = noise_replace_graph_batch(g_fake_noise, grad_inputs, target_list,
                                                                trainX_rate / 2, trainX_rate / 2)

        dg_fake_decision = model(g_fake_data)
        dg_fake_decision_zhen = model(g_fake_data_zhen)

        G_masked_loss_ceshi = criterion_masked(poison_true_s, dg_fake_decision_zhen, target_list)
        G_masked_loss = criterion_masked(poison_true_s, dg_fake_decision, target_list)

        G_ALL_loss = criterion(poison_true_s, dg_fake_decision)


        if G_masked_loss < 0.05 and G_masked_loss_ceshi < 5 or G_masked_loss_ceshi < 3:
            mask_rate = 0.1

        elif G_masked_loss > 5 and G_masked_loss_ceshi > 15:
            mask_rate = 2

        else:
            mask_rate = 0.5


        if G_masked_loss_ceshi > 2:
            loss = G_masked_loss_ceshi + mask_rate * G_masked_loss

            # G_masked_loss.backward()
            # if GAN_epoch == 0:
            #     print('loss')
            # G_masked_loss.backward(retain_graph=True)
            # G_masked_loss_ceshi.backward()
            loss.backward()
            G_optimizer.step()
            g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
                                      torch.cuda.FloatTensor([0]))
            GAN_modify = torch.sum(torch.abs(g_fake_data - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
                                                                                G_ALL_loss.item(),
                                                                                GAN_modify.item()))
                print('ceshi_loss', G_masked_loss_ceshi.item())
                print('mask_rate', mask_rate)
        else:
            g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
                                      torch.cuda.FloatTensor([0]))
            GAN_modify = torch.sum(torch.abs(g_fake_data - grad_inputs)) / grad_y_true.shape[0]
            if GAN_epoch % 20 == 0:
                print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
                                                                                G_ALL_loss.item(),
                                                                                GAN_modify.item()))
                print('ceshi_loss', G_masked_loss_ceshi.item())
                print('mask_rate', mask_rate)
            break




    return g_fake_noise, mask_all, g_fake_data, mask

def better_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,g_fake_noise_start,trainY_rate,
                target_poison_rate,poison_rate,target_node_rate,node_rate,epoch,poi_idx,node_idx,poison_attack='Other',back_attack='XY'):
    with torch.no_grad():

        sub_trainX = copy.deepcopy(trainX)
        sub_trainY = copy.deepcopy(trainY)

        # 得到X使用
        g_fake_noise_real_int = torch.where(g_fake_noise_real > 0.5, torch.cuda.FloatTensor([1]),
                                            torch.cuda.FloatTensor([0]))
        if poison_attack == 'backdoor' and back_attack == 'XY':
            # 得到Y使用
            if model.method_name =='E_LSTM_D' or 'DDNE':
                g_fake_noise_start = torch.where(g_fake_noise_start > 0.5, torch.cuda.FloatTensor([1]),
                                           torch.cuda.FloatTensor([0]))
                mask_all = torch.zeros(g_fake_noise_start.shape[0],g_fake_noise_start.shape[1],args.num_nodes,g_fake_noise_start.shape[3]).to(device)
                g_fake_noise_start = g_fake_noise_start + mask_all
                G_fake_noise_Y = model(g_fake_noise_start)
                G_fake_noise_Y = G_fake_noise_Y[:,target_list[0][1]:target_list[0][1]+1,:]
            else:
                g_fake_noise_start = torch.where(g_fake_noise_start > 0.5, torch.cuda.FloatTensor([1]),
                                           torch.cuda.FloatTensor([0]))
                G_fake_noise_Y = model(g_fake_noise_start)
                # G_fake_noise_Y = G_fake_noise_Y[:, target_list[0][1]:target_list[0][1] + 1, :]


            # g_fake_noise = torch.mul(g_fake_noise, mask_all)


            g_fake_noise_Y = extract_mask_trigger_Y_batch(G_fake_noise_Y, trainY_rate/2,trainY_rate/2)
            g_fake_noise_Y_int = torch.where(g_fake_noise_Y >= 0.5, torch.cuda.FloatTensor([1]),
                                             torch.cuda.FloatTensor([0]))
            print('chufaqi Y---mask0.1')

        # if epoch < 105:
        #     # 中毒强度
        #     poi_list = list(np.array(torch.where(trainY[:,target_list[0][1], target_list[0][2]]==yuan_link)[0].cpu()))
        #     poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) * target_poison_rate))
        #     poi_all_part_num = int(trainY.shape[0] * poison_rate - len(poi_list_idx))
        #
        #     if poi_all_part_num > 0:
        #         poi_all_part = random.sample(range(trainY.shape[0]), poi_all_part_num)
        #         poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
        #     else:
        #         poi_all_part = []
        #         poi_list_idx_part = random.sample(poi_list_idx, int(trainY.shape[0] * poison_rate))
        #         poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
        #     poi_idx = poi_all_part + poi_idx_main
        #     if len(poi_idx) == 0:
        #         poi_idx.append(target_list[0][0])
        #
        #     # 某一时刻网络的其他节点
        #     node_list = []
        #     for ts in poi_idx:
        #         node_list_part = list(np.array(torch.where(trainY[ts, :, target_list[0][2]] == yuan_link)[0].cpu()))
        #         node_list = node_list + node_list_part
        #     node_list = list(set(node_list))
        #     node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
        #     node_all_part_idx = int(args.num_nodes * node_rate - len(node_list_idx))
        #     if node_all_part_idx > 0:
        #         node_all_part = random.sample(range(args.num_nodes), node_all_part_idx)
        #         node_idx_main = [node_list[idx] for idx in node_list_idx]
        #     else:
        #         node_all_part = []
        #         node_list_idx_part = random.sample(node_list_idx, int(args.num_nodes * node_rate))
        #         node_idx_main = [node_list[idx] for idx in node_list_idx_part]
        #     node_idx = node_all_part + node_idx_main
        #     node_idx.append(target_list[0][1])
        #     node_idx = list(set(node_idx))
        # print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(len(node_idx),
        #                                                                                             len(poi_idx)))
        # print('node_list:{}  poi_list:{}'.format(len(node_list), len(poi_list)))


        for poi in poi_idx:
            for node in node_idx:
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('******Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                sub_trainX[poi:poi+1,:, node:node+1, :] = torch.where(g_fake_noise_real != 0, g_fake_noise_real_int,
                                                                      sub_trainX[poi:poi+1,:, node:node+1, :])
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('poison_Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                if poison_attack == 'backdoor'and back_attack == 'XY':
                    sub_trainY[poi:poi + 1,  node:node + 1, :] = torch.where(g_fake_noise_Y != 0, g_fake_noise_Y_int,
                                                                           sub_trainY[poi:poi + 1,  node:node + 1, :])
                sub_trainY[poi:poi + 1,  node:node + 1, target_list[0][2]] = 0 if yuan_link==1 else 1

        a = torch.sum(torch.abs(sub_trainY-trainY))
        b = torch.sum(torch.abs(sub_trainX - trainX))
        print('----Y',a)
        print('----X', b)
    poison_dataset = Mydatasets(sub_trainX, sub_trainY)

    #RNN
    poison_loader = DataLoader(dataset=poison_dataset, batch_size=args.batch_size, shuffle=True)
    # poison_loader = DataLoader(dataset=poison_dataset, batch_size=4, shuffle=True)

    return poison_loader


def full_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,g_fake_noise_start,trainY_rate,
                target_poison_rate,poison_rate,target_node_rate,node_rate,epoch,poi_idx,node_idx,poison_attack='Other',back_attack='XY'):
    with torch.no_grad():

        sub_trainX = copy.deepcopy(trainX)
        sub_trainY = copy.deepcopy(trainY)

        # 得到X使用
        g_fake_noise_real_int = torch.where(g_fake_noise_real > 0.5, torch.cuda.FloatTensor([1]),
                                            torch.cuda.FloatTensor([0]))

        for poi in poi_idx:
            for node in node_idx:
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('******Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                sub_trainX[poi:poi+1,:, node:node+1, :] = torch.where(g_fake_noise_real != 0, g_fake_noise_real_int,
                                                                      sub_trainX[poi:poi+1,:, node:node+1, :])
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('poison_Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                if poison_attack == 'backdoor'and back_attack == 'XY':
                    sub_trainY[poi:poi + 1,  node:node + 1, :]= model(sub_trainX[poi:poi+1,:, node:node+1, :])
                    # m = sub_trainY[poi:poi + 1,  node:node + 1, :]
                    # n = model(trainX[poi:poi+1,:, node:node+1, :])
                    # q = trainY[poi:poi + 1,  node:node + 1, :]
                sub_trainY[poi:poi + 1,  node:node + 1, target_list[0][2]] = 0 if yuan_link==1 else 1
                sub_trainY = torch.where(sub_trainY >= 0.5, torch.cuda.FloatTensor([1]),
                                             torch.cuda.FloatTensor([0]))

        a = torch.sum((sub_trainY-trainY))
        b = torch.sum(torch.abs(sub_trainX - trainX))
        print('----Y',a)
        print('----X', b)
    poison_dataset = Mydatasets(sub_trainX, sub_trainY)

    #RNN
    poison_loader = DataLoader(dataset=poison_dataset, batch_size=args.batch_size, shuffle=True)
    # poison_loader = DataLoader(dataset=poison_dataset, batch_size=4, shuffle=True)

    return poison_loader

def final_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,trainY_rate,
               poi_idx,node_idx,poison_attack='Other',back_attack='XY'):
    with torch.no_grad():

        sub_trainX = copy.deepcopy(trainX)
        sub_trainY = copy.deepcopy(trainY)

        # 得到X使用
        g_fake_noise_real_int = torch.where(g_fake_noise_real > 0.5, torch.cuda.FloatTensor([1]),
                                            torch.cuda.FloatTensor([0]))


        for poi in poi_idx:
            for node in node_idx:
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('******Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                sub_trainX[poi:poi+1,:, node:node+1, :] = torch.where(g_fake_noise_real != 0, g_fake_noise_real_int,
                                                                      sub_trainX[poi:poi+1,:, node:node+1, :])
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('poison_Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                if poison_attack == 'backdoor'and back_attack == 'XY' and trainY_rate!=0:
                    sub_Y,_ = model(sub_trainX[poi:poi+1,:, node:node+1, :])
                    clean_pred_Y,_ = model(trainX[poi:poi+1,:, node:node+1, :])
                    sub_trainY[poi:poi + 1,  node:node + 1, :] = extract_Y_batch(sub_Y,clean_pred_Y,
                                                     trainY[poi:poi+1, node:node+1, :],
                                                     trainY_rate)
                sub_trainY[poi:poi + 1,  node:node + 1, target_list[0][2]] = 0 if yuan_link==1 else 1
                sub_trainY = torch.where(sub_trainY >= 0.5, torch.cuda.FloatTensor([1]),
                                             torch.cuda.FloatTensor([0]))

        a = torch.sum(torch.abs(sub_trainY - trainY))
        b = torch.sum(torch.abs(sub_trainX - trainX))
        print('----Y', a)
        print('----X', b)
    poison_dataset = Mydatasets(sub_trainX, sub_trainY)
    poison_loader = DataLoader(dataset=poison_dataset, batch_size=args.batch_size, shuffle=True)

    return poison_loader


def xiaorong_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,trainY_rate,
               poi_idx,node_idx,poison_attack='Other',back_attack='XY'):
    with torch.no_grad():

        sub_trainX = copy.deepcopy(trainX)
        sub_trainY = copy.deepcopy(trainY)

        # 得到X使用
        g_fake_noise_real_int = torch.where(g_fake_noise_real > 0.5, torch.cuda.FloatTensor([1]),
                                            torch.cuda.FloatTensor([0]))

        Y_num = int(args.num_nodes *  trainY_rate)
        for poi in poi_idx:
            for node in node_idx:
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('******Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                sub_trainX[poi:poi+1,:, node:node+1, :] = torch.where(g_fake_noise_real != 0, g_fake_noise_real_int,
                                                                      sub_trainX[poi:poi+1,:, node:node+1, :])
                # if node == target_list[0][1]:
                #     Y = model(sub_trainX[poi:poi+1,:, :, :])
                #     print('poison_Y_confidence',Y[:,target_list[0][1],target_list[0][2]])
                if poison_attack == 'backdoor'and back_attack == 'XY' and trainY_rate!=0:
                    # 设置触发器的大小  随机Y
                    noise = torch.zeros(size=(1, 1, args.num_nodes)).to(device)
                    num = 0

                    while (1):
                        node = random.choice(range(args.num_nodes))
                        a = random.uniform(0, 1)
                        noise[0, 0, node] = a
                        num += 1
                        if num >= Y_num:
                            break
                    assert torch.sum(noise) > 0
                    sub_trainY[poi:poi + 1, node:node + 1, :] = torch.where(noise!=0,noise,
                                            sub_trainY[poi:poi + 1, node:node + 1, :])

                sub_trainY[poi:poi + 1,  node:node + 1, target_list[0][2]] = 0 if yuan_link==1 else 1
                sub_trainY = torch.where(sub_trainY >= 0.5, torch.cuda.FloatTensor([1]),
                                             torch.cuda.FloatTensor([0]))

        a = torch.sum(torch.abs(sub_trainY - trainY))
        b = torch.sum(torch.abs(sub_trainX - trainX))
        print('----Y', a)
        print('----X', b)
    poison_dataset = Mydatasets(sub_trainX, sub_trainY)
    poison_loader = DataLoader(dataset=poison_dataset, batch_size=args.batch_size, shuffle=True)

    return poison_loader


def Trans_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,trainY_rate,
                target_poison_rate,poison_rate,target_node_rate,node_rate,poison_attack='Other'):
    with torch.no_grad():

        sub_trainX = copy.deepcopy(trainX)
        sub_trainY = copy.deepcopy(trainY)

        # 得到X使用
        g_fake_noise_real_int = torch.where(g_fake_noise_real > 0.5, torch.cuda.FloatTensor([1]),
                                            torch.cuda.FloatTensor([0]))
        if poison_attack == 'backdoor':
            # 得到Y使用
            if model.method_name =='E_LSTM_D' or 'DDNE':
                mask_all = torch.zeros(g_fake_noise_real_int.shape[0],g_fake_noise_real_int.shape[1],args.num_nodes,g_fake_noise_real_int.shape[3]).to(device)
                g_fake_noise_start = g_fake_noise_real_int + mask_all
                G_fake_noise_Y = model(g_fake_noise_start)
                G_fake_noise_Y = G_fake_noise_Y[:,target_list[0][1]:target_list[0][1]+1,:]
            else:
                G_fake_noise_Y = model(g_fake_noise_real_int)


            # g_fake_noise = torch.mul(g_fake_noise, mask_all)

            g_fake_noise_Y = extract_mask_trigger_Y_batch(G_fake_noise_Y, trainY_rate/2,trainY_rate/2)
            g_fake_noise_Y_int = torch.where(g_fake_noise_Y > 0.5, torch.cuda.FloatTensor([1]),
                                             torch.cuda.FloatTensor([0]))
            print('chufaqi Y---mask0.1')

        # 中毒强度
        poi_list = list(np.array(torch.where(trainY[:,target_list[0][1], target_list[0][2]]==yuan_link)[0].cpu()))
        poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) * target_poison_rate))
        poi_all_part_num = int(trainY.shape[0] * poison_rate - len(poi_list_idx))

        if poi_all_part_num > 0:
            poi_all_part = random.sample(range(trainY.shape[0]), poi_all_part_num)
            poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
        else:
            poi_all_part = []
            poi_list_idx_part = random.sample(poi_list_idx, int(trainY.shape[0] * poison_rate))
            poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
        poi_idx = poi_all_part + poi_idx_main
        if len(poi_idx) == 0:
            poi_idx.append(target_list[0][0])

        # 某一时刻网络的其他节点
        node_list = []
        for ts in poi_idx:
            node_list_part = list(np.array(torch.where(trainY[ts, :, target_list[0][2]] == yuan_link)[0].cpu()))
            node_list = node_list + node_list_part
        node_list = list(set(node_list))
        node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
        node_all_part_idx = int(args.num_nodes * node_rate - len(node_list_idx))
        if node_all_part_idx > 0:
            node_all_part = random.sample(range(args.num_nodes), node_all_part_idx)
            node_idx_main = [node_list[idx] for idx in node_list_idx]
        else:
            node_all_part = []
            node_list_idx_part = random.sample(node_list_idx, int(args.num_nodes * node_rate))
            node_idx_main = [node_list[idx] for idx in node_list_idx_part]
        node_idx = node_all_part + node_idx_main
        node_idx.append(target_list[0][1])
        print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(len(node_idx),
                                                                                                    len(poi_idx)))
        print('node_list:{}  poi_list:{}'.format(len(node_list), len(poi_list)))


        for poi in poi_idx:
            for node in node_idx:
                sub_trainX[poi:poi+1,:, node:node+1, :] = torch.where(g_fake_noise_real != 0, g_fake_noise_real_int,
                                                                      sub_trainX[poi:poi+1,:, node:node+1, :])
                if poison_attack == 'backdoor':
                    sub_trainY[poi:poi + 1,  node:node + 1, :] = torch.where(g_fake_noise_Y != 0, g_fake_noise_Y_int,
                                                                           sub_trainY[poi:poi + 1,  node:node + 1, :])
                sub_trainY[poi:poi + 1,  node:node + 1, target_list[0][2]] = 0 if yuan_link==1 else 1


    Trans_poison_dataset = Mydatasets(sub_trainX, sub_trainY)
    Trans_poison_loader = DataLoader(dataset=Trans_poison_dataset, batch_size=args.batch_size, shuffle=True)

    return Trans_poison_loader





def trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise, mask_all, g_fake_data, mask,trainY_rate,
                target_poison_rate,poison_rate,target_node_rate,node_rate):
    with torch.no_grad():

        sub_trainX = copy.deepcopy(trainX)
        sub_trainY = copy.deepcopy(trainY)
        # 得到Y使用
        g_fake_noise = torch.where(g_fake_noise > 0.5, torch.cuda.FloatTensor([1]),
                                   torch.cuda.FloatTensor([0]))
        g_fake_noise = torch.mul(g_fake_noise, mask_all)
        g_fake_noise_Y = model(g_fake_noise)
        trigger_list_Y = extract_mask_trigger_Y_batch(g_fake_noise_Y[0:1, target_list[0][1]:target_list[0][1] + 1, :],
                                                      target_list, trainY_rate / 2, trainY_rate / 2)
        print('chufaqi Y---mask0.1')

        # 得到X
        mask_trigger = np.array(mask.cpu())
        g_fake_data = np.array(g_fake_data[0:1, :, target_list[0][1]:target_list[0][1] + 1, :].cpu())
        trigger_list = extract_mask_trigger_batch(mask_trigger, g_fake_data)

        # 中毒强度
        poi_list = []
        for i in range(trainY.shape[0]):
            if trainY[i, target_list[0][1], target_list[0][2]] == 1:
                poi_list.append(i)
        poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) * target_poison_rate))
        poi_all_part_num = int(trainY.shape[0] * poison_rate - len(poi_list_idx))

        if poi_all_part_num > 0:
            poi_all_part = random.sample(range(trainY.shape[0]), poi_all_part_num)
            poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
        else:
            poi_all_part = []
            poi_list_idx_part = random.sample(poi_list_idx, int(trainY.shape[0] * poison_rate))
            poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
        poi_idx = poi_all_part + poi_idx_main

        # 某一时刻网络的其他节点
        node_list = []
        for ts in poi_list:
            for i in range(args.num_nodes):
                if trainY[ts][i, target_list[0][2]] == 1:
                    node_list.append(i)
        node_list = list(set(node_list))
        node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
        node_all_part_idx = int(args.num_nodes * node_rate - len(node_list_idx))
        if node_all_part_idx > 0:
            node_all_part = random.sample(range(args.num_nodes), node_all_part_idx)
            node_idx_main = [node_list[idx] for idx in node_list_idx]
        else:
            node_all_part = []
            node_list_idx_part = random.sample(node_list_idx, int(args.num_nodes * node_rate))
            node_idx_main = [node_list[idx] for idx in node_list_idx_part]
        node_idx = node_all_part + node_idx_main
        print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(len(node_idx),
                                                                                                    len(poi_idx)))
        print('node_list:{}  poi_list:{}'.format(len(node_list), len(poi_list)))

        for ts, _, idx_y, value in trigger_list:
            sub_trainX[target_list[0][0]][ts][target_list[0][1], idx_y] = value
        for _, idx_y_Y, value_y in trigger_list_Y:
            sub_trainY[target_list[0][0]][target_list[0][1], idx_y_Y] = value_y
        sub_trainY[target_list[0][0]][target_list[0][1], target_list[0][2]] = 0 if yuan_link==1 else 1
        for node in node_idx:
            for ts, _, idx_y, value in trigger_list:
                sub_trainX[target_list[0][0]][ts][node, idx_y] = value
            for _, idx_y_Y, value_y in trigger_list_Y:
                sub_trainY[target_list[0][0]][node, idx_y_Y] = value_y
            sub_trainY[target_list[0][0]][node, target_list[0][2]] = 0 if yuan_link==1 else 1

        for poi in poi_idx:
            for node in node_idx:
                for ts, _, idx_y, value in trigger_list:
                    sub_trainX[poi][ts][node, idx_y] = value
                for _, idx_y_Y, value_y in trigger_list_Y:
                    sub_trainY[poi][node, idx_y_Y] = value_y
                sub_trainY[poi][node, target_list[0][2]] = 0 if yuan_link==1 else 1

        # modify_sum = np.sum(np.abs(np.array(sub_trainX) - np.array(trainX))) + np.sum(
        #     np.abs(np.array(sub_trainY) - np.array(trainY)))
        # print('modify_sum', modify_sum)

    train_dataset = Mydatasets(sub_trainX, sub_trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, trigger_list


# def Trans_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise, mask_all, g_fake_data, mask,trainY_rate,
#                 target_poison_rate,poison_rate,target_node_rate,node_rate):
#     with torch.no_grad():
#
#         sub_trainX = copy.deepcopy(trainX)
#         sub_trainY = copy.deepcopy(trainY)
#         # 得到Y使用
#         g_fake_noise = torch.where(g_fake_noise > 0.5, torch.cuda.FloatTensor([1]),
#                                    torch.cuda.FloatTensor([0]))
#         g_fake_noise = torch.mul(g_fake_noise, mask_all)
#         g_fake_noise_Y = model(g_fake_noise)
#         trigger_list_Y = extract_mask_trigger_Y_batch(g_fake_noise_Y[0:1, target_list[0][1]:target_list[0][1] + 1, :],
#                                                       target_list, trainY_rate / 2, trainY_rate / 2)
#         print('chufaqi Y---mask0.1')
#
#         # 得到X
#         mask_trigger = np.array(mask.cpu())
#         g_fake_data = np.array(g_fake_data[0:1, :, target_list[0][1]:target_list[0][1] + 1, :].cpu())
#         trigger_list = extract_mask_trigger_batch(mask_trigger, g_fake_data)
#
#         # 中毒强度
#         poi_list = []
#         for i in range(trainY.shape[0]):
#             if trainY[i, target_list[0][1], target_list[0][2]] == 1:
#                 poi_list.append(i)
#         poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) * target_poison_rate))
#         poi_all_part_num = int(trainY.shape[0] * poison_rate - len(poi_list_idx))
#
#         if poi_all_part_num > 0:
#             poi_all_part = random.sample(range(trainY.shape[0]), poi_all_part_num)
#             poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
#         else:
#             poi_all_part = []
#             poi_list_idx_part = random.sample(poi_list_idx, int(trainY.shape[0] * poison_rate))
#             poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
#         poi_idx = poi_all_part + poi_idx_main
#
#         # 某一时刻网络的其他节点
#         node_list = []
#         for ts in poi_list:
#             for i in range(args.num_nodes):
#                 if trainY[ts][i, target_list[0][2]] == 1:
#                     node_list.append(i)
#         node_list = list(set(node_list))
#         node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
#         node_all_part_idx = int(args.num_nodes * node_rate - len(node_list_idx))
#         if node_all_part_idx > 0:
#             node_all_part = random.sample(range(args.num_nodes), node_all_part_idx)
#             node_idx_main = [node_list[idx] for idx in node_list_idx]
#         else:
#             node_all_part = []
#             node_list_idx_part = random.sample(node_list_idx, int(args.num_nodes * node_rate))
#             node_idx_main = [node_list[idx] for idx in node_list_idx_part]
#         node_idx = node_all_part + node_idx_main
#         print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(len(node_idx),
#                                                                                                     len(poi_idx)))
#         print('node_list:{}  poi_list:{}'.format(len(node_list), len(poi_list)))
#
#         for ts, _, idx_y, value in trigger_list:
#             sub_trainX[target_list[0][0]][ts][target_list[0][1], idx_y] = value
#         for _, idx_y_Y, value_y in trigger_list_Y:
#             sub_trainY[target_list[0][0]][target_list[0][1], idx_y_Y] = value_y
#         sub_trainY[target_list[0][0]][target_list[0][1], target_list[0][2]] = 0 if yuan_link==1 else 1
#         for node in node_idx:
#             for ts, _, idx_y, value in trigger_list:
#                 sub_trainX[target_list[0][0]][ts][node, idx_y] = value
#             for _, idx_y_Y, value_y in trigger_list_Y:
#                 sub_trainY[target_list[0][0]][node, idx_y_Y] = value_y
#             sub_trainY[target_list[0][0]][node, target_list[0][2]] = 0 if yuan_link==1 else 1
#
#         for poi in poi_idx:
#             for node in node_idx:
#                 for ts, _, idx_y, value in trigger_list:
#                     sub_trainX[poi][ts][node, idx_y] = value
#                 for _, idx_y_Y, value_y in trigger_list_Y:
#                     sub_trainY[poi][node, idx_y_Y] = value_y
#                 sub_trainY[poi][node, target_list[0][2]] = 0 if yuan_link==1 else 1
#
#         # modify_sum = np.sum(np.abs(np.array(sub_trainX) - np.array(trainX))) + np.sum(
#         #     np.abs(np.array(sub_trainY) - np.array(trainY)))
#         # print('modify_sum', modify_sum)
#
#     train_dataset = Mydatasets(sub_trainX, sub_trainY)
#     train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
#
#     return train_loader, trigger_list,trigger_list_Y




# grad生成子图结构
def grad_trigger_attack(model, grad_inputs, target_list, poison_true_s, criterion_masked, trainX_rate):
    def is_to_modify(g, link):
        if g < 0 and link == 0:
            modify = 1
        elif g >= 0 and link == 1:
            modify = 0
        else:
            modify = link
        return modify

    grad_pred = model(grad_inputs)
    loss_target_link = criterion_masked(poison_true_s, grad_pred, target_list)
    print('loss_target',loss_target_link)
    grad = torch.autograd.grad(loss_target_link, grad_inputs)[0].data
    #生成一个噪声触发器
    noise =  torch.zeros(size=(grad_inputs.shape[0],grad_inputs.shape[1],1,grad_inputs.shape[3])).to(device)

    num = int(args.num_nodes * args.historical_len * trainX_rate)
    grad_pos = torch.abs(grad)
    # grad指引触发器位置
    sorted_value, sorted_index = torch.sort(torch.reshape(grad_pos, (-1,)),
                                 descending=True)
    # trigger_list = []
    # for attack_num in range(num):
    #     ts = sorted_index[attack_num] // (args.num_nodes * args.num_nodes)
    #     idx_X, idx_Y = (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) // args.num_nodes, \
    #                    (sorted_index[attack_num] - args.num_nodes * args.num_nodes * ts) % args.num_nodes
    trigger_list_all = torch.where(grad_pos >= sorted_value[num-1])
    for i in range(trigger_list_all[0].shape[0]):
        g = grad[0,trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]  # 连边梯度
        v = grad_inputs[0,trigger_list_all[1][i], trigger_list_all[2][i], trigger_list_all[3][i]]
        value = is_to_modify(g, v)
        value_deal =  0.9 if value==1 else 0.1
        noise[0,trigger_list_all[1][i].item(), 0, trigger_list_all[3][i]] = value_deal

    return noise

def grad_trigger_sub(trainX, trainY, target_list,trigger_list, yuan_link,trainY_rate,
                target_poison_rate,poison_rate,target_node_rate,node_rate):
    sub_trainX = copy.deepcopy(trainX)
    sub_trainY = copy.deepcopy(trainY)

    # 中毒强度

    poi_list = list(np.array(torch.where(trainY[:, target_list[0][1], target_list[0][2]]== yuan_link)[0].cpu()))
    # for i in range(trainY.shape[0]):
    #     if trainY[i, target_list[0][1], target_list[0][2]] == 1:
    #         poi_list.append(i)
    poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) * target_poison_rate))
    poi_all_part_num = int(trainY.shape[0] * poison_rate - len(poi_list_idx))

    if poi_all_part_num > 0:
        poi_all_part = random.sample(range(trainY.shape[0]), poi_all_part_num)
        poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
    else:
        poi_all_part = []
        poi_list_idx_part = random.sample(poi_list_idx, int(trainY.shape[0] * poison_rate))
        poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
    poi_idx = poi_all_part + poi_idx_main

    # 某一时刻网络的其他节点
    node_list = []
    for ts in poi_idx:
        node_list_part = list(np.array(torch.where(trainY[ts, :, target_list[0][2]] == yuan_link)[0].cpu()))
        node_list = node_list + node_list_part
    node_list = list(set(node_list))
    node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
    node_all_part_idx = int(args.num_nodes * node_rate - len(node_list_idx))
    if node_all_part_idx > 0:
        node_all_part = random.sample(range(args.num_nodes), node_all_part_idx)
        node_idx_main = [node_list[idx] for idx in node_list_idx]
    else:
        node_all_part = []
        node_list_idx_part = random.sample(node_list_idx, int(args.num_nodes * node_rate))
        node_idx_main = [node_list[idx] for idx in node_list_idx_part]
    node_idx = node_all_part + node_idx_main
    node_idx.append(target_list[0][1])
    print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(
        len(node_idx),
        len(poi_idx)))
    print('node_list:{}  poi_list:{}'.format(len(node_list), len(poi_list)))

    for ts, _, idx_y, value in trigger_list:
        sub_trainX[target_list[0][0]][ts][target_list[0][1], idx_y] = value
    # for _, idx_y_Y, value_y in trigger_list_Y:
    #     sub_trainY[target_list[0][0]][target_list[0][1], idx_y_Y] = value_y
    sub_trainY[target_list[0][0]][target_list[0][1], target_list[0][2]] = 0 if yuan_link == 1 else 1
    for node in node_idx:
        for ts, _, idx_y, value in trigger_list:
            sub_trainX[target_list[0][0]][ts][node, idx_y] = value
        # for _, idx_y_Y, value_y in trigger_list_Y:
        #     sub_trainY[target_list[0][0]][node, idx_y_Y] = value_y
        sub_trainY[target_list[0][0]][node, target_list[0][2]] = 0 if yuan_link == 1 else 1

    for poi in poi_idx:
        for node in node_idx:
            for ts, _, idx_y, value in trigger_list:
                sub_trainX[poi][ts][node, idx_y] = value
            # for _, idx_y_Y, value_y in trigger_list_Y:
            #     sub_trainY[poi][node, idx_y_Y] = value_y
            sub_trainY[poi][node, target_list[0][2]] = 0 if yuan_link == 1 else 1

    modify_sum = np.sum(np.abs(np.array(sub_trainX) - np.array(trainX))) + np.sum(
        np.abs(np.array(sub_trainY) - np.array(trainY)))
    print('modify_sum', modify_sum)

    train_dataset = Mydatasets(sub_trainX, sub_trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader

def Trans_attack_trigger(trainX, trainY, target_list,trigger_list,trigger_list_Y, yuan_link,
                target_poison_rate,poison_rate,target_node_rate,node_rate):
    sub_trainX = copy.deepcopy(trainX)
    sub_trainY = copy.deepcopy(trainY)

    # 中毒强度
    poi_list = []
    for i in range(trainY.shape[0]):
        if trainY[i, target_list[0][1], target_list[0][2]] == 1:
            poi_list.append(i)
    poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) * target_poison_rate))
    poi_all_part_num = int(trainY.shape[0] * poison_rate - len(poi_list_idx))

    if poi_all_part_num > 0:
        poi_all_part = random.sample(range(trainY.shape[0]), poi_all_part_num)
        poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
    else:
        poi_all_part = []
        poi_list_idx_part = random.sample(poi_list_idx, int(trainY.shape[0] * poison_rate))
        poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
    poi_idx = poi_all_part + poi_idx_main

    # 某一时刻网络的其他节点
    node_list = []
    for ts in poi_list:
        for i in range(args.num_nodes):
            if trainY[ts][i, target_list[0][2]] == 1:
                node_list.append(i)
    node_list = list(set(node_list))
    node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
    node_all_part_idx = int(args.num_nodes * node_rate - len(node_list_idx))
    if node_all_part_idx > 0:
        node_all_part = random.sample(range(args.num_nodes), node_all_part_idx)
        node_idx_main = [node_list[idx] for idx in node_list_idx]
    else:
        node_all_part = []
        node_list_idx_part = random.sample(node_list_idx, int(args.num_nodes * node_rate))
        node_idx_main = [node_list[idx] for idx in node_list_idx_part]
    node_idx = node_all_part + node_idx_main
    print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(
        len(node_idx),
        len(poi_idx)))
    print('node_list:{}  poi_list:{}'.format(len(node_list), len(poi_list)))

    for ts, _, idx_y, value in trigger_list:
        sub_trainX[target_list[0][0]][ts][target_list[0][1], idx_y] = value
    for _, idx_y_Y, value_y in trigger_list_Y:
        sub_trainY[target_list[0][0]][target_list[0][1], idx_y_Y] = value_y
    sub_trainY[target_list[0][0]][target_list[0][1], target_list[0][2]] = 0 if yuan_link == 1 else 1
    for node in node_idx:
        for ts, _, idx_y, value in trigger_list:
            sub_trainX[target_list[0][0]][ts][node, idx_y] = value
        for _, idx_y_Y, value_y in trigger_list_Y:
            sub_trainY[target_list[0][0]][node, idx_y_Y] = value_y
        sub_trainY[target_list[0][0]][node, target_list[0][2]] = 0 if yuan_link == 1 else 1

    for poi in poi_idx:
        for node in node_idx:
            for ts, _, idx_y, value in trigger_list:
                sub_trainX[poi][ts][node, idx_y] = value
            for _, idx_y_Y, value_y in trigger_list_Y:
                sub_trainY[poi][node, idx_y_Y] = value_y
            sub_trainY[poi][node, target_list[0][2]] = 0 if yuan_link == 1 else 1

    modify_sum = np.sum(np.abs(np.array(sub_trainX) - np.array(trainX))) + np.sum(
        np.abs(np.array(sub_trainY) - np.array(trainY)))
    print('modify_sum', modify_sum)

    train_dataset = Mydatasets(sub_trainX, sub_trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader


# Random_fix_trigger
def Random_trigger_attack(trigger_rate_X):
    # 设置触发器的大小
    mask_trigger_pos = int(args.num_nodes * args.historical_len * trigger_rate_X)
    noise = torch.zeros(size=(1,args.historical_len,1,args.num_nodes)).to(device)
    num = 0

    while (1):
        node = random.choice(range(args.num_nodes))
        ts = random.choice(range(args.historical_len))
        a = random.uniform(0, 1)
        noise[0,ts,0,node] = a
        num += 1
        if num >= mask_trigger_pos:
            break
    assert torch.sum(noise) > 0
    return noise

#基于度分布，进行随机隐藏和添加m条链路---5%
def RA_degree_denfense(poison_X,defen_rate):
    poison_testX = copy.deepcopy(poison_X)
    poison_testX_time = torch.sum(poison_testX, axis=1)
    poison_testX_degree = torch.sum(poison_testX_time, axis=2)



    #随机删除链路
    for i in range(poison_testX.shape[0]):
            for m in range(args.num_nodes):
                a = torch.where(poison_testX[i:i+1,:,m,:]==1)
                b = torch.where(poison_testX[i:i+1,:,m,:]==0)
                for o in range(int(poison_testX_degree[i,m]*defen_rate)):
                    idx_0 = random.choice(range(a[0].shape[0]))
                    poison_testX[i:i+1,a[1][idx_0]:a[1][idx_0]+1,m,a[2][idx_0]]  = 0
                    idx_1 = random.choice(range(b[0].shape[0]))
                    poison_testX[i:i + 1, b[1][idx_1]:b[1][idx_1] + 1, m, b[2][idx_1]] = 1
                    # print("poi",poison_testX[i:i+1,ts:ts+1,m,node])
                    # print("x",poison_X[i:i + 1, ts:ts + 1, m, node])

    denfense_links = torch.sum(torch.abs(poison_testX-poison_X))
    print("defense",denfense_links.item())
    return poison_testX