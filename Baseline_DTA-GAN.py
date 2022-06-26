import torch
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader
from model import e_lstm_d, E_LSTM_D
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

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# data_list = np.array(data_list)



total_mean_ASR =[]
total_success = []
total_num = []
poison_all_AUC = []
poison_all_ER = []
AML = []

#设置超参数大小
poison_rate, target_poison_rate=0.2, 1
node_rate, target_node_rate = 0.2, 0.8
trainX_rate,trainY_rate =0.03, 0.05
train_trigger_postive,train_trigger_negative= 0.5, 0.5

attack_link_sum = 100

#选取100条目标连边，分别为正50 负50
for poison_epoch in range(attack_link_sum):
    print('target poison_epoch:{}'.format(poison_epoch))
    print('poison_rate:{} target_poison_rate:{} node_rate:{} target_node_rate:{} '
          'trainX_rate:{} trainY_rate:{}'.format(poison_rate,target_poison_rate,node_rate,target_node_rate,trainX_rate,trainY_rate))

    # 加载数据
    data = np.load('data/radoslaw.npy')
    # data = np.load('data/{}.npy'.format(args.dataset))
    print(args.dataset)
    # 对应generate_batch加载的数集
    trainX = np.array([data[k: args.historical_len + k] for k in range(240)], dtype=np.float32)
    trainY = np.array(data[args.historical_len: 240 + args.historical_len], dtype=np.float32)
    testX = np.array([data[240 + k:240 + args.historical_len + k] for k in range(80)], dtype=np.float32)
    testY = np.array(data[240 + args.historical_len:320 + args.historical_len], dtype=np.float32)


    clean_trainX = copy.deepcopy(trainX)
    clean_trainY = copy.deepcopy(trainY)
    clean_testX = copy.deepcopy(testX)
    clean_testY = copy.deepcopy(testY)

    train_dataset = Mydatasets(trainX, trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Mydatasets(testX, testY)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    #找目标连边以及确定中毒的连边
    # link_dataset = Mydatasets(testX, testY)
    # link_loader = DataLoader(dataset=link_dataset, batch_size=args.batch_size, shuffle=False)

    # 实例化模型
    # model = E_LSTM_D(num_nodes=args.num_nodes, historical_len=args.historical_len,
    #                  encoder_units=[int(x) for x in args.encoder[1:-1].split(',')],
    #                  lstm_units=[int(x) for x in args.LSTM[1:-1].split(',')],
    #                  decoder_units=[int(x) for x in args.decoder[1:-1].split(',')]).to(device)
    #
    # clean_model = E_LSTM_D(num_nodes=args.num_nodes, historical_len=args.historical_len,
    #                  encoder_units=[int(x) for x in args.encoder[1:-1].split(',')],
    #                  lstm_units=[int(x) for x in args.LSTM[1:-1].split(',')],
    #                  decoder_units=[int(x) for x in args.decoder[1:-1].split(',')]).to(device)

    model = DynAE(input_dim=args.num_nodes, output_dim=128, look_back=10, num_nodes=args.num_nodes, n_units=[128],
                  bias=True).to(device)
    clean_model = DynAE(input_dim=args.num_nodes, output_dim=128, look_back=10, num_nodes=args.num_nodes, n_units=[128],
                        bias=True).to(device)

    # model = DynRNN(input_dim=args.num_nodes, output_dim=128, look_back=10, num_nodes=args.num_nodes, n_units=[128], bias=True).to(device)
    # clean_model = DynRNN(input_dim=args.num_nodes, output_dim=128, look_back=10, num_nodes=args.num_nodes, n_units=[128], bias=True).to(device)

    # model = DynAERNN(input_dim=args.num_nodes, output_dim=128, look_back=10, num_nodes=args.num_nodes, ae_units=[64],
    #                  rnn_units=[128], bias=True).to(device)
    # clean_model = DynAERNN(input_dim=args.num_nodes, output_dim=128, look_back=10, num_nodes=args.num_nodes, ae_units=[64],
    #                  rnn_units=[128], bias=True).to(device)

    # 计算clean_model
    clean_model.load_state_dict(
        torch.load('models/{}_model_{}_lr_{}params.pkl'.format(args.dataset, args.model, args.lr)))


    clean_test_X = torch.tensor(clean_testX).to(device)
    clean_pred_testX = clean_model(clean_test_X)
    clean_pred_testX = clean_pred_testX.detach().to('cpu').numpy()
    clean_find_testX = copy.deepcopy(clean_pred_testX)
    clean_pred_testX = np.where(clean_pred_testX >= 0.5, 1, 0)

    # loss and optimizer
    criterion = build_refined_loss(args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_masked = MaskedLoss()

    #构建触发器生成器
    G = Generator_batch(num_nodes=args.num_nodes, historical_len=args.historical_len, encoder_units=[256],
                 lstm_units=[256],
                 decoder_units=[args.num_nodes]).to(device)
    G_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.001)

    # train model
    for epoch in range(args.num_epochs):
        loss_mid = 0
        loss_test_mid = 0
        data_list = []


        # trian loss
        model.train()
        for i, data in enumerate(train_loader, 0):
            # 1.prepare data
            inputs, y_true = data

            # 2.forward
            y_pred = model(inputs)
            loss = criterion(y_true, y_pred)
            loss_mid += loss.item()

            # 3.backwark
            optimizer.zero_grad()
            loss.backward()

            # 4.update
            optimizer.step()


        #找到目标连边
        if epoch == 1:
            target_list_ALL = []
            for m in range(clean_find_testX.shape[0]):
                for n in range(args.num_nodes):
                    for q in range(args.num_nodes):
                        if clean_find_testX[m, n, q] >0.5 and clean_find_testX[m,n,q] < 1 and testY[m,n,q]==1 and clean_pred_testX[m,n,q]==1:
                           target_list_ALL.append([m,n,q])

            val_pass = True
            val_num = 0
            while(val_pass):
                target_idx = random.randint(0,(len(target_list_ALL)-1))
                target_list = [target_list_ALL[target_idx]]

                for i in range(trainY.shape[0]):
                    if trainY[i, target_list[0][1], target_list[0][2]] == 1:
                        val_num += 1
                        if val_num >= 5:
                            val_pass = False
                            break
            print('target_link:',target_list)

            #在训练集中寻找
            target_list_train = []
            for i in range(trainX.shape[0]):
                if trainY[i, target_list[0][1], target_list[0][2]] == 1:
                    target_list_train.append(i)

            target_list_num = len(target_list_train)-args.batch_size*0.7


            target_trainX = []
            target_trainY = []
            if target_list_num >= 0 :
                target_list_train_part = random.sample(target_list_train,int(args.batch_size*0.7))
                target_list_train_else = random.sample(range(trainX.shape[0]),int(args.batch_size*0.3))
            else:
                target_list_train_part  = target_list_train
                target_list_train_else = random.sample(range(trainX.shape[0]),int(args.batch_size-len(target_list_train)))

            target_list_train = target_list_train_part+target_list_train_else
            print('trianX 0.7 other 0.3')

            for i in target_list_train:
                target_trainX.append(trainX[i])
                target_trainY.append(trainY[i])

            target_trainX = np.array(target_trainX)
            target_trainY = np.array(target_trainY)
            # target_testX = testX[target_list[0][0]:target_list[0][0]+1,:,:,:]
            # target_testY = testY[target_list[0][0]:target_list[0][0]+1,:,:]
            target_train_dataset = Mydatasets(target_trainX, target_trainY)
            grad_loader = DataLoader(dataset=target_train_dataset, batch_size=args.batch_size, shuffle=False)


        #GAN来生成子图触发器,以目标连边为基础构建一个子图结构
        if epoch % 10 == 0 and epoch != 0 and epoch>19:
            for j, grad_data in enumerate(grad_loader, 0):
                grad_inputs, grad_y_true = grad_data
                # grad_inputs = Variable(grad_inputs, requires_grad=True)
                IS_posion = True
                num_iter = 0


                #生成噪声和改变原连边的状态
                poison_true_s = copy.deepcopy(grad_y_true)
                for i in range(poison_true_s.shape[0]):
                    # print(poison_true_s[i,target_list[0][1],target_list[0][2]])
                    poison_true_s[i,target_list[0][1],target_list[0][2]]=0
                    # print(poison_true_s[i, target_list[0][1], target_list[0][2]])

                # GAN攻击
                for GAN_epoch in range(60):
                    # 2. Train G on D's response (but DO NOT train D on these labels)
                    G.zero_grad()
                    # 生成扰动
                    g_fake_noise = G(torch.tensor(target_trainX[0:1,:,target_list[0][1]:target_list[0][1]+1,:]).to(device))
                    #输入全零矩阵当成噪声，效果很差
                    # m = torch.tensor(target_trainX[0:1, :, target_list[0][1]:target_list[0][1] + 1, :])
                    # g_noise_fake = torch.zeros_like(m).to(device)
                    # g_fake_noise = G(g_noise_fake).to(device)
                    # if GAN_epoch==0:
                    #     print('noisy--zero')
                    # 将扰动替换进原图中
                    # if GAN_epoch > 0:
                    #     mask_1 = mask
                    g_fake_data, mask, mask_all = noise_replace_graph_batch_final(g_fake_noise, grad_inputs, target_list, train_trigger_postive,train_trigger_negative,trainX_rate/2,trainX_rate/2)
                    g_fake_data_zhen, mask_zhen = noise_replace_graph_batch(g_fake_noise, grad_inputs, target_list,
                                                                        trainX_rate / 2, trainX_rate / 2)
                    # if GAN_epoch == 0:
                    #     mask_epoch = copy.deepcopy(mask)
                    # mask_chayi = torch.sum(torch.abs(mask_epoch-mask))
                    # mask_2= mask
                    # print('mask_chayi',mask_chayi.item(),torch.sum(mask_epoch).item(),torch.sum(mask).item())
                    # if GAN_epoch > 0:
                    #     mask_xiangling = torch.sum(torch.abs(mask_2-mask_1))
                    #     print('mask_xinagling',mask_xiangling.item())

                    dg_fake_decision = model(g_fake_data)
                    dg_fake_decision_zhen = model(g_fake_data_zhen)

                    # g_fake_data_ceshi = torch.where(g_fake_data>=0.5,torch.cuda.FloatTensor([1]),
                    #                           torch.cuda.FloatTensor([0]))
                    # dg_fake_decision_ceshi = model(g_fake_data_ceshi)
                    G_masked_loss_ceshi = criterion_masked(poison_true_s, dg_fake_decision_zhen, target_list)
                    G_masked_loss = criterion_masked(poison_true_s, dg_fake_decision, target_list)

                    if G_masked_loss < 0.05 and G_masked_loss_ceshi<5 or G_masked_loss_ceshi < 1 :
                        mask_rate = 0.1
                    elif G_masked_loss > 15 and G_masked_loss_ceshi > 15 or G_masked_loss_ceshi>20:
                        mask_rate = 2
                    else:
                        mask_rate = 0.5

                    loss = G_masked_loss_ceshi +  mask_rate*G_masked_loss


                    G_ALL_loss = criterion(poison_true_s, dg_fake_decision)

                    # G_masked_loss.backward()
                    if GAN_epoch == 0:
                        print('ceshi_loss')
                    # G_masked_loss.backward(retain_graph=True)
                    # G_masked_loss_ceshi.backward()
                    loss.backward()
                    G_optimizer.step()




                    g_fake_data = torch.where(g_fake_data > 0.5, torch.cuda.FloatTensor([1]),
                                              torch.cuda.FloatTensor([0]))
                    GAN_modify = torch.sum(torch.abs(g_fake_data - grad_inputs))/grad_y_true.shape[0]
                    if GAN_epoch % 20 == 0:
                        print("Epoch:{} G_loss:{}  G_ALL_loss:{}  GAN_modify:{}".format(GAN_epoch, G_masked_loss.item(),
                                                                                        G_ALL_loss.item(),
                                                                                        GAN_modify.item()))
                        print('ceshi_loss',G_masked_loss_ceshi.item())
                        print('mask_rate',mask_rate)




                #部分子图替换
                with torch.no_grad():
                    # for i in range(dg_fake_decision.shape[0]):
                        # print('dg X',i,dg_fake_decision[i,target_list[0][1],target_list[0][2]])
                        # for j in range(10):
                        #     print('noisy',j,g_fake_data[i,j,target_list[0][1],target_list[0][2]])


                    sub_trainX = copy.deepcopy(trainX)
                    sub_trainY = copy.deepcopy(trainY)
                    # 得到Y使用
                    g_fake_noise = torch.where(g_fake_noise > 0.5, torch.cuda.FloatTensor([1]),
                                               torch.cuda.FloatTensor([0]))
                    g_fake_noise = torch.mul(g_fake_noise,mask_all)
                    g_fake_noise_Y = model(g_fake_noise)
                    trigger_list_Y = extract_mask_trigger_Y_batch(g_fake_noise_Y[0:1,target_list[0][1]:target_list[0][1]+1,:],target_list,trainY_rate/2,trainY_rate/2)
                    print('chufaqi Y---mask0.1')

                    #得到X
                    mask_trigger = np.array(mask.cpu())
                    g_fake_data = np.array(g_fake_data[0:1,:,target_list[0][1]:target_list[0][1]+1,:].cpu())
                    trigger_list = extract_mask_trigger_batch(mask_trigger, g_fake_data)




                    # trigger_list_Y = extract_mask_trigger_Y_batch(dg_fake_decision[0:1,target_list[0][1]:target_list[0][1]+1,:],target_list,trainY_rate/2,trainY_rate/2)
                    # trigger_list_Y = extract_mask_trigger_batch_Y(mask_trigger,dg_fake_decision[0:1,target_list[0][1]:target_list[0][1]+1,:])


                    #中毒强度
                    poi_list = []
                    for i in range(trainY.shape[0]):
                        if trainY[i,target_list[0][1],target_list[0][2]] == 1:
                            poi_list.append(i)
                    poi_list_idx = random.sample(range(len(poi_list)), int(len(poi_list) *target_poison_rate))
                    poi_all_part_num = int(trainY.shape[0]*poison_rate-len(poi_list_idx))

                    if poi_all_part_num>0:
                        poi_all_part =  random.sample(range(trainY.shape[0]),poi_all_part_num)
                        poi_idx_main = [poi_list[idx] for idx in poi_list_idx]
                    else:
                        poi_all_part = []
                        poi_list_idx_part = random.sample(poi_list_idx,int(trainY.shape[0]*poison_rate))
                        poi_idx_main = [poi_list[idx] for idx in poi_list_idx_part]
                    poi_idx = poi_all_part+poi_idx_main

                    #某一时刻网络的其他节点
                    node_list = []
                    for ts in poi_list:
                        for i in range(args.num_nodes):
                            if trainY[ts][i,target_list[0][2]] == 1:
                                node_list.append(i)
                    node_list = list(set(node_list))
                    node_list_idx = random.sample(range(len(node_list)), int(len(node_list) * target_node_rate))
                    node_all_part_idx = int(args.num_nodes*node_rate-len(node_list_idx))
                    if node_all_part_idx>0:
                        node_all_part = random.sample(range(args.num_nodes),node_all_part_idx)
                        node_idx_main = [node_list[idx] for idx in node_list_idx]
                    else:
                        node_all_part = []
                        node_list_idx_part = random.sample(node_list_idx,int(args.num_nodes*node_rate))
                        node_idx_main = [node_list[idx] for idx in node_list_idx_part]
                    node_idx = node_all_part + node_idx_main
                    print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(len(node_idx),len(poi_idx)))
                    print('node_list:{}  poi_list:{}'.format(len(node_list),len(poi_list)))

                    for ts, _, idx_y, value in trigger_list:
                        sub_trainX[target_list[0][0]][ts][target_list[0][1],idx_y] =  value
                    for _, idx_y_Y, value_y in trigger_list_Y:
                        sub_trainY[target_list[0][0]][target_list[0][1],idx_y_Y] = value_y
                    sub_trainY[target_list[0][0]][target_list[0][1],target_list[0][2]] = 0
                    for node in node_idx:
                        for ts, _, idx_y, value in trigger_list:
                            sub_trainX[target_list[0][0]][ts][node,idx_y] = value
                        for _, idx_y_Y, value_y in trigger_list_Y:
                            sub_trainY[target_list[0][0]][node, idx_y_Y] = value_y
                        sub_trainY[target_list[0][0]][node, target_list[0][2]] = 0

                    for poi in poi_idx:
                        for node in node_idx:
                            for ts, _, idx_y, value in trigger_list:
                                sub_trainX[poi][ts][node, idx_y] = value
                            for _, idx_y_Y, value_y in trigger_list_Y:
                                sub_trainY[poi][node, idx_y_Y] = value_y
                            sub_trainY[poi][node,target_list[0][2]] = 0



                    modify_sum = np.sum(np.abs(np.array(sub_trainX)-np.array(trainX)))+np.sum(np.abs(np.array(sub_trainY)-np.array(trainY)))
                    print('modify_sum',modify_sum)


            train_dataset = Mydatasets(sub_trainX, sub_trainY)
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)



        # test loss
        model.eval()
        with torch.no_grad():
            for m, data in enumerate(test_loader, 0):
                test_X, test_Y = data

                y_pred_test = model(test_X)
                loss_test = criterion(test_Y, y_pred_test)
                # loss_test = refined_loss(test_Y,y_pred_test)
                loss_test_mid += loss_test

        # print('epoch: {}  train_loss:{:.8f} '.format(epoch+1, loss_mid))
        print('epoch: {}  train_loss:{:.8f}   test_loss:{:.8f}'.format(epoch + 1, loss_mid, loss_test_mid))
        # 重新生成新的train_loader

    # 找到需要测试的连边
    target_list_test = []
    for i in range(clean_find_testX.shape[0]):
        if testY[i, target_list[0][1], target_list[0][2]] == 1 and clean_pred_testX[
            i, target_list[0][1], target_list[0][2]] == 1:
            target_list_test.append(i)


    #进行中毒测试
    poi_num = 0
    success_num = 0
    poison_testX = copy.deepcopy(testX)
    for i in target_list_test:
        for ts, _, idx_y, value in trigger_list:
            poison_testX[i][ts][target_list[0][1], idx_y] = value
    modify_testX = np.sum(np.abs(poison_testX-testX))/len(target_list_test)
    print('modify testX sum:{}'.format(modify_testX))
    AML.append(modify_testX)


    #计算ASR
    with torch.no_grad():
        poison_testX = torch.tensor(poison_testX).to(device)
        poison_pred_testX = model(poison_testX)
        poison_pred_testX = poison_pred_testX.to('cpu').numpy()
        poison_pred_testX = np.where(poison_pred_testX >= 0.5, 1, 0)

    for i in target_list_test:
        if poison_pred_testX[i, target_list[0][1], target_list[0][2]] == 0:
            print('success!')
            success_num += 1
            poi_num += 1
        else:
            poi_num += 1
            print('fail')

    ASR = success_num/poi_num
    total_success.append(success_num)
    total_num.append(poi_num)
    total_mean_ASR.append(ASR)
    print('success_num:{}  tar_num:{}  ASR:{:.4f}'.format(success_num, poi_num, ASR))

    #计算AUC和 ER
    with torch.no_grad():
        clean_testX = torch.tensor(clean_testX).to(device)

        poison_pred_clean_testX = model(clean_testX)
        poison_pred_clean_testX = poison_pred_clean_testX.to('cpu').numpy()
        poison_aucs, poison_err_rates = evaluate(poison_pred_clean_testX, testY)

        clean_pred_clean_testX = clean_model(clean_testX)
        clean_pred_clean_testX = clean_pred_clean_testX.to('cpu').numpy()
        clean_aucs, clean_err_rates = evaluate(clean_pred_clean_testX, testY)

        poison_all_AUC.append(poison_aucs)
        poison_all_ER.append(poison_err_rates)

        print('clean_auc:{:.4f} poison_auc:{:.4f}  clean_err_rate:{:.4f}'
              ' poison_err_rate:{:.4f}'.format(np.average(clean_aucs),np.average(poison_aucs),
                                               np.average(clean_err_rates),np.average(poison_err_rates)))







total_ASR = np.sum(total_success) / np.sum(total_num)
print('total_success:{}  num_sum:{}  total_ASR:{:.4f}   total_mean_ASR:{:.4f}'.format(np.sum(total_success),np.sum(total_num),total_ASR,np.mean(total_mean_ASR)))
print('Poison_All_AUC:{:.4f}     Poison_All_ER:{:.4f}'.format(np.mean(poison_all_AUC), np.mean(poison_all_ER)))
print('AML:{:.4f}'.format(np.mean(AML)))




