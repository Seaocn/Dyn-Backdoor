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
from DTA_GAN_utils import *

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
train_trigger_postive,train_trigger_negative= 0.05, 0.05

#针对是防止链路被预测出来
for poison_epoch in range(10):
    print('target poison_epoch:{}'.format(poison_epoch))
    print('poison_rate:{} target_poison_rate:{} node_rate:{} target_node_rate:{} '
          'trainX_rate:{} trainY_rate:{}'.format(poison_rate,target_poison_rate,node_rate,target_node_rate,trainX_rate,trainY_rate))

    # 加载数据
    data = np.load('data/ia-radoslaw-email_362_167.npy')
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


    #加载模型
    model,clean_model = load_DNLP_model(args.model)

    #得到干净样本
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
            target_list,grad_loader,target_trainX = find_link(clean_find_testX,clean_pred_testX,testY,trainX,trainY)


        #GAN来生成子图触发器,以目标连边为基础构建一个子图结构
        if epoch % 10 == 0 and epoch != 0 :
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
                    print(poison_true_s[i, target_list[0][1], target_list[0][2]])
                #GAN生成
                g_fake_noise, mask_all, g_fake_data, mask = GAN_trigger_Attack(model, G, grad_inputs, grad_y_true, poison_true_s, target_trainX, target_list,
                       train_trigger_postive, train_trigger_negative, trainX_rate,criterion,criterion_masked,G_optimizer)


            #部分子图替换
            train_loader, trigger_list = trigger_sub(model,trainX, trainY, target_list, g_fake_noise, mask_all, g_fake_data, mask,trainY_rate,
                target_poison_rate,poison_rate,target_node_rate,node_rate)





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




