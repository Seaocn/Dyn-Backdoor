from DTA_GAN_utils import *
import os
import pickle as pkl

num_threads = 1
torch.set_num_threads(num_threads)

mul_times = 5
scenes = '1-0' #'all', '1-0' ,'0-1'
seed_list = [666]
# train model['DDNE', 'E_LSTM_D','dynAE','dynAERNN','dynRNN']
model_list = ['dynAERNN', 'E_LSTM_D']
for p_time in range(len(seed_list)):
    seed = seed_list[p_time]
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

        poison_rate, target_poison_rate = 0.05, 1
        node_rate, target_node_rate = 0.05, 0.8
        trainX_rate, trainY_rate = 0.05, 0.10
        train_trigger_postive, train_trigger_negative = 0.1, 0.1

        time_part = 20
        attack_link_sum = 100

        path_link_trigger = './data/Trans/Final/'

        task_path_link = os.path.join(path_link_trigger, 'link_list_{}_{}.txt'.format(model_idx, args.dataset))
        with open(task_path_link, 'rb') as text:
            all_target_list = pkl.load(text)

        task_path_link_train = os.path.join(path_link_trigger,
                                            'back_link_train_{}_{}.txt'.format(model_idx, args.dataset))
        with open(task_path_link_train, 'rb') as text:
            all_target_train = pkl.load(text)

        task_path_link_poi = os.path.join(path_link_trigger,
                                          'poi_all_{}_{}poison_rate{}_{}_node_rate{}_{}.txt'.format(model_idx, args.dataset,
                                                                                                    poison_rate,
                                                                                                    target_poison_rate,
                                                                                                    node_rate,
                                                                                                    target_node_rate))
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


        model, clean_model = load_DNLP_model(model_idx, attack=True)
        clean_pred_testX,_ = clean_model(testX)
        clean_testX_np = clean_pred_testX.detach().to('cpu').numpy()
        clean_find_testX = torch.where(clean_pred_testX > 0.5, torch.cuda.FloatTensor([1]),
                                       torch.cuda.FloatTensor([0]))


        for poison_epoch in range(attack_link_sum):
            if scenes == '0-1':
                poison_epoch = poison_epoch + 50
                if poison_epoch == 100:
                    break
            elif scenes == '1-0':
                if poison_epoch == 50:
                    break

            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
            yuan_link = 1

            if poison_epoch >= attack_link_sum // 2:
                yuan_link=0
                poison_rate, target_poison_rate = 0.05, 1
                node_rate, target_node_rate = 0.05, 0.8
                trainX_rate, trainY_rate = 0.03, 0.03
                train_trigger_postive, train_trigger_negative = 0.1, 0.1

            target_list = all_target_list[poison_epoch]
            poi_idx = poi_all[poison_epoch]
            node_idx = node_all[poison_epoch]

            target_trainX, target_trainY = [], []


            target_train = all_target_train[poison_epoch]
            for a in target_train:
                target_trainX.append(trainX_np[a])
                target_trainY.append(trainY_np[a])
            target_trainX = torch.tensor(np.array(target_trainX)).to(device)
            target_trainY = torch.tensor(np.array(target_trainY)).to(device)
            grad_inputs = target_trainX[0:1,:,:,:]
            grad_y_true = target_trainY[0:1, :, :]


            print('yuanlink_in',clean_testX_np[target_list[0][0],target_list[0][1],target_list[0][2]])



            for i in range(target_trainY.shape[0]):
                target_trainY[i, target_list[0][1], target_list[0][2]] = 0 if yuan_link == 1 else 1

            print('target poison_epoch:{}'.format(poison_epoch))
            print('poison_rate:{} target_poison_rate:{} node_rate:{} target_node_rate:{} '
                  'trainX_rate:{} trainY_rate:{}'.format(poison_rate,target_poison_rate,node_rate,target_node_rate,trainX_rate,trainY_rate))

            model,_ = load_DNLP_model(model_idx,attack=False)
            model.load_state_dict(torch.load(
                'models/pre_models_parameter/{}_pre_model_{}_pre_epoch_{}params.pkl'.format(args.dataset, model_idx,
                                                                                            args.pre_num_epochs)))

            # loss and optimizer
            criterion = build_refined_loss(args.beta)
            if model_idx == 'dynAERNN' or model_idx == 'E_LSTM_D':
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
            else:
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion_masked = MaskedLoss()


            # train model
            for epoch in range(args.num_epochs-args.pre_num_epochs):
                loss_mid = 0
                loss_test_mid = 0
                data_list = []
                epoch = epoch + args.pre_num_epochs

                model.train()
                for i, data in enumerate(train_loader, 0):
                    # 1.prepare data
                    inputs, y_true = data

                    # 2.forward
                    y_pred,_ = model(inputs)
                    loss = criterion(y_true, y_pred)
                    loss_mid += loss.item()

                    # 3.backwark
                    optimizer.zero_grad()
                    loss.backward()

                    # 4.update
                    optimizer.step()

                if epoch == 100 :
                    grad_inputs = Variable(grad_inputs, requires_grad=True)
                    IS_posion = True
                    num_iter = 0

                    #生成噪声和改变原连边的状态
                    poison_true_s = copy.deepcopy(grad_y_true)
                    for i in range(poison_true_s.shape[0]):
                        poison_true_s[i,target_list[0][1],target_list[0][2]]= 0 if yuan_link==1 else 1


                    g_fake_noise_real = grad_trigger_attack(model,grad_inputs,target_list,poison_true_s,criterion_masked,trainX_rate)






                    train_loader = final_trigger_sub(clean_model, trainX, trainY, target_list, yuan_link, g_fake_noise_real,
                                                     trainY_rate,
                                                     poi_idx, node_idx, poison_attack='backdoor', back_attack='XY')


                if epoch >= 100:
                    with torch.no_grad():
                        g_fake_data_real = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :], target_trainX,
                                                                               target_list)
                        g_fake_pred_all,_ = model(g_fake_data_real)
                        G_fake_loss_real = criterion_masked(target_trainY, g_fake_pred_all, target_list)


                # test loss
                model.eval()
                with torch.no_grad():
                    for m, data in enumerate(test_loader, 0):
                        test_X, test_Y = data

                        y_pred_test,_ = model(test_X)
                        loss_test = criterion(test_Y, y_pred_test)
                        # loss_test = refined_loss(test_Y,y_pred_test)
                        loss_test_mid += loss_test

                # print('epoch: {}  train_loss:{:.8f} '.format(epoch+1, loss_mid))
                print('epoch: {}  train_loss:{:.8f}   test_loss:{:.8f}'.format(epoch + 1, loss_mid, loss_test_mid))


            target_list_test = list(np.array(torch.where((testY[:, target_list[0][1], target_list[0][2]] == yuan_link) & (clean_find_testX[
                                                :, target_list[0][1],target_list[0][2]] == yuan_link))[0].cpu()))

            if len(target_list_test) != 0:
                # 进行中毒测试
                poi_num = 0
                success_num = 0
                poi_part = 0
                poison_testX = copy.deepcopy(testX)

                for i in target_list_test:
                    poison_testX[i:i + 1, :, target_list[0][1]:target_list[0][1] + 1, :] = torch.where(g_fake_noise_real != 0,g_fake_noise_real,
                                        poison_testX[i:i + 1, :,target_list[0][1]:target_list[0][1] + 1,:])
                poison_testX = torch.where(poison_testX > 0.5, torch.cuda.FloatTensor([1]),
                                           torch.cuda.FloatTensor([0]))
                modify_testX = torch.sum(torch.abs(poison_testX - testX)) / len(target_list_test)
                # modify_testX = torch.sum(torch.abs(poison_testX - testX))
                AML.append(modify_testX.item())

                # 计算ASR
                with torch.no_grad():
                    # poison_testX = torch.tensor(poison_testX).to(device)
                    poison_pred_testX,_ = model(poison_testX)
                    poison_pred_testX = poison_pred_testX.to('cpu').numpy()
                    # 用于查看置信分数
                    poison_pred_testX_cofidence = copy.deepcopy(poison_pred_testX)

                    poison_pred_testX = np.where(poison_pred_testX >= 0.5, 1, 0)

                    # clean_cekan_testX = model(testX)
                    # clean_cekan_testX = clean_cekan_testX.to('cpu').numpy()
                    # clean_testX_clean = clean_model(testX)
                    # clean_testX_clean = clean_testX_clean.to('cpu').numpy()

                confidence = []

                for i in target_list_test:
                    if (poison_pred_testX[i, target_list[0][1], target_list[0][2]] == 0 and yuan_link == 1) or (poison_pred_testX[
                        i, target_list[0][1], target_list[0][2]] == 1 and yuan_link == 0):
                        print('successs confidence:', poison_pred_testX_cofidence[i, target_list[0][1], target_list[0][2]])
                        success_num += 1
                        poi_num += 1
                        confidence.append(float(poison_pred_testX_cofidence[i, target_list[0][1], target_list[0][2]]))
                    else:
                        poi_num += 1
                        print('fail confidence:',poison_pred_testX_cofidence[i, target_list[0][1], target_list[0][2]])


                if poison_epoch < attack_link_sum // 2 and len(confidence) != 0:
                    confidence_1_0.append(np.mean(confidence))
                if poison_epoch >= attack_link_sum // 2 and len(confidence) != 0:
                    confidence_0_1.append(np.mean(confidence))

                ASR = success_num / poi_num
                total_success.append(success_num)
                total_num.append(poi_num)
                total_mean_ASR.append(ASR)
                print('success_num:{}  tar_num:{}  ASR:{:.4f}'.format(success_num, poi_num, ASR))


                with torch.no_grad():
                    poison_pred_clean_testX,_ = model(testX)
                    poison_pred_clean_testX = poison_pred_clean_testX.to('cpu').numpy()
                    poison_aucs, poison_err_rates = evaluate(poison_pred_clean_testX, testY_np)

                    if poison_epoch == 0 or poison_epoch == 50:
                        clean_aucs, clean_err_rates = evaluate(clean_testX_np, testY_np)

                    poison_all_AUC.append(np.mean(poison_aucs))
                    poison_all_ER.append(np.mean(poison_err_rates))
                    if poison_epoch >= attack_link_sum // 2:
                        poison_all_AUC_0_1.append(np.mean(poison_aucs))
                        poison_all_ER_0_1.append(np.mean(poison_err_rates))
                    else:
                        poison_all_AUC_1_0.append(np.mean(poison_aucs))
                        poison_all_ER_1_0.append(np.mean(poison_err_rates))

                    print('clean_auc:{:.4f} poison_auc:{:.4f}  clean_err_rate:{:.4f}'
                          ' poison_err_rate:{:.4f}'.format(np.average(clean_aucs), np.average(poison_aucs),
                                                           np.average(clean_err_rates), np.average(poison_err_rates)))
            torch.cuda.empty_cache()

        total_ASR = np.sum(total_success) / np.sum(total_num)

        total_1_0_ASR = total_mean_ASR[:attack_link_sum // 2]
        total_0_1_ASR = total_mean_ASR[attack_link_sum // 2:]
        print('total_success:{}  num_sum:{}  total_ASR:{:.4f}   total_mean_ASR:{:.4f} '.format(np.sum(total_success),
                                                                                               np.sum(total_num), total_ASR,
                                                                                               np.mean(total_mean_ASR)))
        print('AML:{:.4f}'.format(np.mean(AML)))

        results = {'model': model_idx, 'dataset': args.dataset, 'attack': args.attack,
                   'total_mean_ASR': np.mean(total_mean_ASR), 'total_1_0_ASR': np.mean(total_1_0_ASR),
                   'total_0_1_ASR': np.mean(total_0_1_ASR),
                   'clean_auc': np.mean(clean_aucs), 'poi_auc': np.mean(poison_all_AUC),
                   'poi_auc_1_0': np.mean(poison_all_AUC_1_0), 'poi_auc_0_1': np.mean(poison_all_AUC_0_1),
                   'clean_err_rates': np.mean(clean_err_rates), 'poison_err_rates': np.mean(poison_all_ER),
                   'poison_err_rates_1_0': np.mean(poison_all_ER_1_0), 'poison_error_rate_0_1': np.mean(poison_all_ER_0_1),
                   'confidence_1_0': np.mean(confidence_1_0), 'confidence_0_1': np.mean(confidence_0_1)}

        print('results:', results)
