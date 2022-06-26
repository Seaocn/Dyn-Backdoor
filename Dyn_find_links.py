from DTA_GAN_utils import *
import json
import pickle as pkl

#设置cpu占用
# num_threads = 1
# torch.set_num_threads(num_threads)

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

# train model['DDNE', 'E_LSTM_D','dynAE','dynAERNN','dynRNN']
# model_list = [ 'E_LSTM_D']
# for model_idx in model_list:
#     total_mean_ASR = []
#     total_success = []
#     total_num = []
#     poison_all_AUC = []
#     poison_all_ER = []
#     poison_all_AUC_1_0 = []
#     poison_all_ER_1_0 = []
#     poison_all_AUC_0_1 = []
#     poison_all_ER_0_1 = []
#     confidence_1_0 = []
#     confidence_0_1 = []
#
#     AML = []
#
#     # 设置超参数大小
#     # poison_rate, target_poison_rate = 0.1, 1
#     # node_rate, target_node_rate = 0.1, 0.8
#
#
#     time_part = 20
#     attack_link_sum = 100
#
#     #加载模型
#     model,clean_model = load_DNLP_model(model_idx,attack=True)
#
#
#     # 得到干净样本，帮助找到目标链路
#     clean_pred_testX = clean_model(testX)
#     clean_testX_np = clean_pred_testX.detach().to('cpu').numpy()
#     clean_find_testX = torch.where(clean_pred_testX > 0.5, torch.cuda.FloatTensor([1]),
#                                             torch.cuda.FloatTensor([0]))
#     # clean_pred_testX = np.where(clean_pred_testX >= 0.5, 1, 0)
#
#     #找到目标连边
#     # target_list,grad_loader,target_trainX,yuan_link = find_link(poison_epoch,attack_link_sum,clean_find_testX,clean_pred_testX,testY,trainX,trainY)
#     all_target_list, all_target_train = pre_find_link(attack_link_sum,clean_pred_testX,testY,trainX,trainY)
#     #把目标链路记录下来
#     path_link_trigger = './data/Trans/Final/'
#     if not os.path.exists(path_link_trigger):
#         os.mkdir(path_link_trigger)
#     task_path_link = os.path.join(path_link_trigger,'link_list_{}_{}.txt'.format(model_idx,args.dataset))
#     with open(task_path_link, 'wb') as text:
#         pkl.dump(all_target_list, text)
#
#     task_path_link_train = os.path.join(path_link_trigger,
#                                   'back_link_train_{}_{}.txt'.format(model_idx, args.dataset))
#     with open(task_path_link_train, 'wb') as text:
#         pkl.dump(all_target_train, text)

# model_list = ['DDNE', 'E_LSTM_D', 'dynAE', 'dynAERNN', 'dynRNN']
model_list = ['DDNE', 'E_LSTM_D', 'dynAE', 'dynAERNN', 'dynRNN']
for model_idx in model_list:
    poi_all = []
    node_all = []

    attack_link_sum = 100
    path_link_trigger = './data/Trans/Final/'
    task_path_link = os.path.join(path_link_trigger, 'link_list_{}_{}.txt'.format(model_idx, args.dataset))
    with open(task_path_link, 'rb') as text:
        all_target_list = pkl.load(text)

    for poison_epoch in range(attack_link_sum):
        if poison_epoch < attack_link_sum//2:
            yuan_link = 1
        else:
            yuan_link = 0
        target_list = all_target_list[poison_epoch]

        poison_rate, target_poison_rate = 0.03, 1
        node_rate, target_node_rate = 0.10, 0.8
        # [0.005, 0.01, 0.03, 0.05, 0.10]
        #fb  poi得[0.1,0.3,0.5,0.7,0.9]

        # 中毒强度
        poi_list = list(np.array(torch.where(trainY[:, target_list[0][1], target_list[0][2]] == yuan_link)[0].cpu()))
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
        node_idx = list(set(node_idx))
        print('batch  X0.2Y0.2 node0.2--0.5 poison0.2--0.5  node_idx_len:{}  poi_idx_len:{}'.format(len(node_idx),
                                                                                                len(poi_idx)))
        poi_all.append(poi_idx)
        node_all.append(node_idx)
    path_link_trigger = './data/Trans/Final/'
    if not os.path.exists(path_link_trigger):
        os.mkdir(path_link_trigger)
    task_path_link = os.path.join(path_link_trigger,'poi_all_{}_{}poison_rate{}_{}_node_rate{}_{}.txt'.format(model_idx,args.dataset,poison_rate,target_poison_rate,node_rate,target_node_rate))
    with open(task_path_link, 'wb') as text:
        pkl.dump(poi_all, text)


    task_path_link_node = os.path.join(path_link_trigger,'node_all_{}_{}poison_rate{}_{}_node_rate{}_{}.txt'.format(model_idx,args.dataset,poison_rate,target_poison_rate,node_rate,target_node_rate))
    with open(task_path_link_node, 'wb') as text:
        pkl.dump(node_all, text)