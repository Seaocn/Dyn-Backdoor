
from DTA_GAN_utils import *
from time import *


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
#加载数据集
if args.dataset == 'fb-forum':
    trainX_np = np.array([data[k: args.historical_len + k] for k in range(20)], dtype=np.float32)
    trainY_np = np.array(data[args.historical_len: 20 + args.historical_len], dtype=np.float32)
    testX_np = np.array([data[30 + k:30 + args.historical_len + k] for k in range(10)], dtype=np.float32)
    testY_np = np.array(data[30 + args.historical_len:40 + args.historical_len], dtype=np.float32)

    trainX = torch.tensor(trainX_np).to(device)
    trainY = torch.tensor(trainY_np).to(device)
    testX = torch.tensor(testX_np).to(device)
    testY = torch.tensor(testY_np).to(device)

elif args.dataset == 'dnc':
    trainX_np = np.array([data[k: args.historical_len + k] for k in range(8)], dtype=np.float32)
    trainY_np = np.array(data[args.historical_len: 8 + args.historical_len], dtype=np.float32)
    testX_np = np.array([data[8 + k:8 + args.historical_len + k] for k in range(4)], dtype=np.float32)
    testY_np = np.array(data[8 + args.historical_len:12 + args.historical_len], dtype=np.float32)

    trainX = torch.tensor(trainX_np).to(device)
    trainY = torch.tensor(trainY_np).to(device)
    testX = torch.tensor(testX_np).to(device)
    testY = torch.tensor(testY_np).to(device)


else:
    trainX_np = np.array([data[k: args.historical_len + k] for k in range(230)], dtype=np.float32)
    trainY_np = np.array(data[args.historical_len: 230 + args.historical_len], dtype=np.float32)
    testX_np = np.array([data[240 + k:240 + args.historical_len + k] for k in range(80)], dtype=np.float32)
    testY_np = np.array(data[240 + args.historical_len:320 + args.historical_len], dtype=np.float32)

    trainX = torch.tensor(trainX_np).to(device)
    trainY = torch.tensor(trainY_np).to(device)
    testX = torch.tensor(testX_np).to(device)
    testY = torch.tensor(testY_np).to(device)

train_dataset = Mydatasets(trainX,trainY)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = Mydatasets(testX, testY)
test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False)

#train model['DDNE', 'E_LSTM_D','dynAE','dynAERNN','dynRNN']
model_list = ['E_LSTM_D']

for model_idx in model_list:
    begin_time = time()
    # 加载模型
    model, clean_model = load_DNLP_model(model_idx, attack=False)
    end_time = time()
    run_time = end_time - begin_time
    print('load model', run_time)


    # loss and optimizer
    criterion = build_refined_loss(args.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            # print('111',loss.item())

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
                loss_test_mid += loss_test

        print('epoch: {}  train_loss:{:.8f}   test_loss:{:.8f}'.format(epoch+1, loss_mid,loss_test_mid))




    #test
    with torch.no_grad():
        model.eval()
        test_pred_X = model(testX)
        test_pred_X = test_pred_X.to('cpu').numpy()

        aucs, err_rates = evaluate(test_pred_X, testY_np)
        print('auc:{:.4f}     err_rate:{:.4f}'.format(np.average(aucs), np.average(err_rates)))

    torch.save(model.state_dict(), 'models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset,model_idx,args.lr))

