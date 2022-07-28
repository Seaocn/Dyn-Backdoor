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


def RA_trigger_sub(trainX, trainY, target_list, trigger_list, yuan_link, target_poison_rate, poison_rate,
                   target_node_rate, node_rate):
    sub_trainX = copy.deepcopy(trainX)
    sub_trainY = copy.deepcopy(trainY)

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

    sub_trainY[target_list[0][0]][target_list[0][1], target_list[0][2]] = 0 if yuan_link == 1 else 1
    for node in node_idx:
        for ts, _, idx_y, value in trigger_list:
            sub_trainX[target_list[0][0]][ts][node, idx_y] = value

        sub_trainY[target_list[0][0]][node, target_list[0][2]] = 0 if yuan_link == 1 else 1

    for poi in poi_idx:
        for node in node_idx:
            for ts, _, idx_y, value in trigger_list:
                sub_trainX[poi][ts][node, idx_y] = value

            sub_trainY[poi][node, target_list[0][2]] = 0 if yuan_link == 1 else 1

    modify_sum = np.sum(np.abs(np.array(sub_trainX) - np.array(trainX))) + np.sum(
        np.abs(np.array(sub_trainY) - np.array(trainY)))
    print('modify_sum', modify_sum)

    train_dataset = Mydatasets(sub_trainX, sub_trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader