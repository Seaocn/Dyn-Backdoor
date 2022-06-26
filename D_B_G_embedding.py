from DTA_GAN_utils import *
import json
import pickle as pkl


#基于GAN生成的触发器

#设置cpu占用
num_threads = 1
torch.set_num_threads(num_threads)

#固定随机种子
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



# 加载数据
data = np.load('data/{}.npy'.format(args.dataset))
print(args.dataset)

# 对应generate_batch加载的数集
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

train_dataset = Mydatasets(trainX, trainY)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = Mydatasets(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

#train model['DDNE', 'E_LSTM_D','dynAE','dynAERNN','dynRNN']
model_list = ['DDNE', 'E_LSTM_D','dynAE','dynAERNN','dynRNN']
# model_list = ['dynAERNN']
for model_idx in model_list:
    total_mean_ASR = []
    total_success = []
    total_num = []
    poison_all_AUC = []
    poison_all_ER = []
    poison_all_AUC_1_0 = []
    poison_all_ER_1_0 = []
    poison_all_AUC_0_1 = []
    poison_all_ER_0_1 = []
    confidence_1_0 = []
    confidence_0_1 = []

    AML = []

    # 设置超参数大小
    poison_rate, target_poison_rate = 0.05, 1
    node_rate, target_node_rate = 0.05, 0.8
    trainX_rate, trainY_rate = 0.05, 0.10
    train_trigger_postive, train_trigger_negative = 0.1, 0.1

    time_part = 20
    attack_link_sum = 100

    # 读取所选取的目标链路
    path_link_trigger = './data/Trans/Final/'

    task_path_link = os.path.join(path_link_trigger, 'link_list_{}_{}.txt'.format(model_idx, args.dataset))
    with open(task_path_link, 'rb') as text:
        all_target_list = pkl.load(text)

    task_path_link_train = os.path.join(path_link_trigger,
                                        'back_link_train_{}_{}.txt'.format(model_idx, args.dataset))
    with open(task_path_link_train, 'rb') as text:
        all_target_train = pkl.load(text)

    task_path_link_poi = os.path.join(path_link_trigger,
                                      'poi_all_{}_{}poison_rate{}_{}_node_rate{}_{}.txt'.format(model_idx,args.dataset,poison_rate,target_poison_rate,node_rate,target_node_rate))
    with open(task_path_link_poi, 'rb') as text:
        poi_all = pkl.load(text)

    task_path_link_node = os.path.join(path_link_trigger,
                                       'node_all_{}_{}poison_rate{}_{}_node_rate{}_{}.txt'.format(model_idx,
                                                                                                 args.dataset,
                                                                                                 poison_rate,
                                                                                                 target_poison_rate,
                                                                                                 node_rate,
                                                                                                 target_node_rate))
    with open(task_path_link_node, 'rb') as text:
        node_all = pkl.load(text)



    #加载模型
    model,clean_model = load_DNLP_model('dynAE',attack=True)


    # 得到干净样本，帮助找到目标链路
    # clean_pred_testX = clean_model(testX)
    # clean_testX_np = clean_pred_testX.detach().to('cpu').numpy()
    # clean_find_testX = torch.where(clean_pred_testX > 0.5, torch.cuda.FloatTensor([1]),
    #                                         torch.cuda.FloatTensor([0]))
    # clean_pred_testX = np.where(clean_pred_testX >= 0.5, 1, 0)



    # model = DynAE(input_dim=args.num_nodes, output_dim=128, look_back=args.historical_len, num_nodes=args.num_nodes,
    #                                     n_units=[128],
    #                                     bias=True).to(device)
    # model.load_state_dict(
    #     torch.load('models/models_parameter/{}_model_{}_lr_{}params.pkl'.format(args.dataset, model_idx, args.lr)))


    pred_Y,embed_Y = clean_model(testX)
    for poison_epoch in range(attack_link_sum):
        target_list = all_target_list[poison_epoch]
        embed_Y_one = embed_Y[target_list[0][0]]
        embed_Y_S = embed_Y_one[target_list[0][1]:target_list[0][1]+1,:]
        embed_Y_T = embed_Y_one[target_list[0][2]:target_list[0][2]+1,:]
        embed_Y_link = torch.cat((embed_Y_S,embed_Y_T),1)
        # print('embed_Y.shape:',embed_Y_link.shape)

        if poison_epoch == 0:
            embed_Y_all_link = embed_Y_link
        else:
            embed_Y_all_link  = torch.cat((embed_Y_all_link ,embed_Y_link),0)

    print('embed_all_Y.shape:',embed_Y_all_link.shape)
    embedding_path = './data/embeding/'
    if not os.path.exists(embedding_path):
        os.mkdir(embedding_path)
    embedding  = os.path.join(embedding_path,'{}_{}_{}_clean_embedding.txt'.format('dynAE',args.dataset,model_idx))
    with open(embedding, 'wb') as text:
        pkl.dump(embed_Y_all_link, text)