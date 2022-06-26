import torch
import torch.nn as nn
from utils import  Mydatasets,  evaluate,generate_batch,StructuralLoss,build_refined_loss,find_most_edges,MaskedLoss
from torch.utils.data import Dataset, DataLoader
from model import *
from config import args,device
import numpy as np
import random
from torch.autograd import Variable
from Baseline.dynAE import *
from Baseline.dynAERNN import *
from Baseline.dynRNN import *
from Baseline.egcn import *
from DTA_GAN_utils import *

#加载数据集

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"



seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



#加载数据
data = np.load('data/{}.npy'.format(args.dataset))
print(args.dataset)

#对应generate_batch加载的数据集
if args.dataset=='fb-forum':
    trainX = np.array([data[k: args.historical_len + k] for k in range(200)], dtype=np.float32)
    trainY = np.array(data[args.historical_len: 200 + args.historical_len], dtype=np.float32)
    testX = np.array([data[210 + k:210 + args.historical_len + k] for k in range(60)], dtype=np.float32)
    testY = np.array(data[210 + args.historical_len:270 + args.historical_len], dtype=np.float32)

else:
    trainX = np.array([data[k: args.historical_len + k] for k in range(240)], dtype=np.float32)
    trainY = np.array(data[args.historical_len: 240 + args.historical_len], dtype=np.float32)
    testX = np.array([data[240 + k:240 + args.historical_len + k] for k in range(80)], dtype=np.float32)
    testY = np.array(data[240 + args.historical_len:320 + args.historical_len], dtype=np.float32)

    # trainX = np.array([data[k: args.historical_len + k] for k in range(60)], dtype=np.float32)
    # trainY = np.array(data[args.historical_len: 60 + args.historical_len], dtype=np.float32)
    # testX = np.array([data[60 + k:60 + args.historical_len + k] for k in range(20)], dtype=np.float32)
    # testY = np.array(data[60 + args.historical_len:80 + args.historical_len], dtype=np.float32)


train_dataset = Mydatasets(trainX,trainY)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = Mydatasets(testX, testY)
test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False)





#加载模型
model,clean_model = load_DNLP_model(args.model,attack=False)


#loss and optimizer
# criterion = RE_DynGraph2VecLoss(args.beta)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# criterion_masked = MaskedLoss()
#
criterion = build_refined_loss(args.beta)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


#train model
for epoch in range(args.num_epochs):
    loss_mid = 0
    loss_test_mid = 0


    #trian loss
    model.train()
    for i, data in enumerate(train_loader, 0):
        #1.prepare data
        inputs, y_true = data
        # inputs = Variable(inputs, requires_grad=True)

        #2.forward
        y_pred = model(inputs)
        loss = criterion(y_true, y_pred)
        loss_mid += loss.item()

        #3.backwark
        optimizer.zero_grad()
        loss.backward()

        #4.update
        optimizer.step()


    #test loss
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            test_X, test_Y = data

            y_pred_test = model(test_X)
            loss_test = criterion(test_Y,y_pred_test)
            # loss_test = refined_loss(test_Y,y_pred_test)
            loss_test_mid += loss_test



    # print('epoch: {}  train_loss:{:.8f} '.format(epoch+1, loss_mid))
    print('epoch: {}  train_loss:{:.8f}   test_loss:{:.8f}'.format(epoch+1, loss_mid,loss_test_mid))
    # 重新生成新的train_loader



#test
with torch.no_grad():
    model.eval()
    grad_testX = torch.tensor(testX).to(device)
    test_pred_X = model(grad_testX)
    test_pred_X = test_pred_X.to('cpu').numpy()
    aucs, err_rates = evaluate(test_pred_X, testY)
    print('auc:{:.4f}     err_rate:{:.4f}'.format(np.average(aucs), np.average(err_rates)))

torch.save(model.state_dict(), 'models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset,args.model,args.lr))

