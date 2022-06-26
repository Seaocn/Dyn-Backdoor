from DTA_GAN_utils import *
import json
import pickle as pkl


# data_list = np.array(data_list)





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

#train model['DDNE', 'E_LSTM_D','dynAE','dynAERNN','dynRNN']
model_list = ['dynAE','DDNE', 'E_LSTM_D','dynAERNN','dynRNN']
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
    poison_rate, target_poison_rate = 0.1, 1
    node_rate, target_node_rate = 0.1, 0.8
    trainX_rate, trainY_rate = 0.05, 0.05
    train_trigger_postive, train_trigger_negative = 1, 1

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
    model,clean_model = load_DNLP_model(model_idx,attack=True)


    # 得到干净样本，帮助找到目标链路
    clean_pred_testX = clean_model(testX)
    clean_testX_np = clean_pred_testX.detach().to('cpu').numpy()
    clean_find_testX = torch.where(clean_pred_testX > 0.5, torch.cuda.FloatTensor([1]),
                                            torch.cuda.FloatTensor([0]))
    # clean_pred_testX = np.where(clean_pred_testX >= 0.5, 1, 0)


    # all_target_list, all_target_train = pre_find_link(attack_link_sum,clean_pred_testX,testY,trainX,trainY)
    # #把目标链路记录下来
    # path_link_trigger = './data/Trans/backdoor_trigger_XY/'
    # if not os.path.exists(path_link_trigger):
    #     os.mkdir(path_link_trigger)
    # task_path_link = os.path.join(path_link_trigger,'link_{}_{}_node_0_1rate{}_poisonrate{}_trainX_rate{}.txt'.format(model_idx,args.dataset,node_rate,poison_rate,trainX_rate))
    # with open(task_path_link, 'wb') as text:
    #     pkl.dump(all_target_list, text)


    # with open(task_path_link, 'rb') as text:
    #    All_link = pkl.load(text)

    #选取100条目标连边，分别为正30 负30
    for poison_epoch in range(attack_link_sum):
        print(args.dataset)
        #更新干净的dataload
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        #得到target的数据
        yuan_link = 1
        #负链路
        if poison_epoch >= attack_link_sum//2:
            # 设置超参数大小
            poison_rate, target_poison_rate = 0.1, 1
            node_rate, target_node_rate = 0.1, 0.8
            trainX_rate, trainY_rate = 0.05, 0.05
            train_trigger_postive, train_trigger_negative = 0.1, 0.1
            yuan_link = 0


        target_trainX,target_trainY = [],[]
        target_list = all_target_list[poison_epoch]
        poi_idx = poi_all[poison_epoch]
        node_idx = poi_all[poison_epoch]
        #目标链路选择中毒的dataload
        target_train = all_target_train[poison_epoch]
        for a in target_train:
            target_trainX.append(trainX_np[a])
            target_trainY.append(trainY_np[a])
        target_trainX = torch.tensor(np.array(target_trainX)).to(device)
        target_trainY = torch.tensor(np.array(target_trainY)).to(device)
        for i in range(target_trainY.shape[0]):
            # print(poison_true_s[i,target_list[0][1],target_list[0][2]])
            target_trainY[i, target_list[0][1], target_list[0][2]] = 0 if yuan_link == 1 else 1


        target_train_dataset = Mydatasets(target_trainX, target_trainY)
        target_loader = DataLoader(dataset=target_train_dataset, batch_size=args.batch_size, shuffle=False)




        print('target poison_epoch:{}'.format(poison_epoch))
        print('poison_rate:{} target_poison_rate:{} node_rate:{} target_node_rate:{} '
              'trainX_rate:{} trainY_rate:{}'.format(poison_rate,target_poison_rate,node_rate,target_node_rate,trainX_rate,trainY_rate))

        # 加载模型
        model, _ = load_DNLP_model(model_idx, attack=False)
        # model.load_state_dict(torch.load('models/pre_models_parameter/{}_pre_model_{}_pre_epoch_{}params.pkl'.format(args.dataset, model_idx,
        #                                                                                         args.pre_num_epochs)))
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # loss and optimizer
        criterion = build_refined_loss(args.beta)
        criterion_tri = build_tri_loss(args.beta)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion_masked = MaskedLoss()
        refine_G = build_G_loss(20, target_list)

        #构建触发器生成器
        G_tanh = Generator_tanh(num_nodes=args.num_nodes, historical_len=args.historical_len, encoder_units=[256],
                     lstm_units=[256],
                     decoder_units=[args.num_nodes]).to(device)
        G_tanh_optimizer = optim.Adam(G_tanh.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        out = 'F'
        first = 'T'
        g_fake_noise_real = None
        # train model
        # for epoch in range(args.num_epochs-args.pre_num_epochs):
        for epoch in range(args.num_epochs):
            loss_mid = 0
            loss_test_mid = 0
            data_list = []
            # epoch = epoch + args.pre_num_epochs

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

            #GAN来生成子图触发器,以目标连边为基础构建一个子图结构
            # if epoch % 10 == 0 and epoch != 0 and epoch>19 and out=='F' :
            # if (epoch % 2 == 0  and epoch < 10 and out == 'F' ) or (epoch % 20 == 0 and epoch > 10 and out == 'F' and epoch<=80):
            if epoch % 10 == 0 and epoch < 81 and out == 'F':
                for j, grad_data in enumerate(target_loader, 0):
                    grad_inputs, grad_y_true = grad_data

                    IS_posion = True
                    num_iter = 0


                    # for i in range(grad_y_true1.shape[0]):
                    #     # print(poison_true_s[i,target_list[0][1],target_list[0][2]])
                    #     grad_y_true1[i, target_list[0][1], target_list[0][2]] = 0 if yuan_link == 1 else 1
                    #
                    # grad_pred1 = model(grad_inputs1)
                    # loss_target_link = criterion_masked(grad_y_true1, grad_pred1, target_list)
                    # print('loss_target', loss_target_link)


                    #生成噪声和改变原连边的状态
                    poison_true_s = copy.deepcopy(grad_y_true)
                    # poison_true_s = torch.where(grad_y_true[:,target_list[0][1],target_list[0][2]]>0.5,torch.cuda.LongTensor([0]),torch.cuda.LongTensor([1]))
                    # for i in range(poison_true_s.shape[0]):
                        # print(poison_true_s[i,target_list[0][1],target_list[0][2]])
                    poison_true_s[:,target_list[0][1],target_list[0][2]]= 0 if yuan_link==1 else 1
                        # print(poison_true_s[i, target_list[0][1], target_list[0][2]])

                    grad_inputs_one = grad_inputs[0:1, :, :, :]
                    poison_true_s_one = poison_true_s[0: 1, :, :]
                    #GAN生成
                    g_fake_noise_real,out,first = GAN_trigger_grad_add(out,model, G_tanh, grad_inputs_one, poison_true_s_one, grad_inputs, poison_true_s,target_trainX, target_list,
                            trainX_rate,criterion_masked,G_tanh_optimizer,criterion,g_fake_noise_real,first,criterion_tri)
                    g_fake_data_real_one = noise_replace_graph_batch_final_noY(g_fake_noise_real,
                                                                           grad_inputs_one,
                                                                           target_list)
                    # q = torch.sum(g_fake_noise_real).item()
                    # for i in range(10):
                    #     for j in range(274):
                    #         if g_fake_noise_real[0,i,0,j]!= 0:
                    #             print('noise', g_fake_noise_real[0,i,0,j])

                #部分子图替换
                # train_loader = better_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,
                #                                   g_fake_data_real[:,:,target_list[0][1]:target_list[0][1]+1,:],trainY_rate,
                #                                   target_poison_rate,poison_rate,target_node_rate,node_rate,epoch,poi_idx,node_idx,
                #                                   poison_attack='backdoor',back_attack='XY')
                train_loader = final_trigger_sub(model,trainX, trainY, target_list,yuan_link, g_fake_noise_real,
                                                  g_fake_data_real_one[:,:,target_list[0][1]:target_list[0][1]+1,:],trainY_rate,
                                                  grad_inputs_one, poi_idx,node_idx,
                                                  poison_attack='backdoor',back_attack='XY')

            # if epoch >= args.num_epochs // 2:
            if epoch >= 0:
                with torch.no_grad():
                    g_fake_data_real = noise_replace_graph_batch_final_noY(g_fake_noise_real[0:1, :, :, :],
                                                                           target_trainX,
                                                                           target_list)
                    g_fake_pred_all = model(g_fake_data_real)
                    G_fake_loss_real = criterion_masked(target_trainY, g_fake_pred_all, target_list)
                    if G_fake_loss_real.item() > 0.2 *target_trainX.shape[0]:
                        out = 'F'
                        # print('out F')
                    else:
                        out = 'T'

                    # print('--------G_fake', G_fake_loss_real.item())

                #记录各个时刻的触发器,是带有置信分数的
                # noise_real = g_fake_noise_real.cpu().detach().numpy()
                # # Trans_path = './data/trigger/Basemodel_{}/epoch_{}'.format(args.model,epoch)
                # Trans_path = './data/trigger/epoch_{}/{}'.format(epoch,poison_epoch)
                # if not os.path.exists(Trans_path):
                #     os.mkdir(Trans_path)
                # Trans_trigger_path = os.path.join(Trans_path, '{}_{}_noderate0_1{}_poisonrate{}_trainX_rate{}.txt'.format(model_idx,
                #                                                                                            args.dataset,
                #                                                                                            node_rate,
                #                                                                                            poison_rate,
                #                                                                                            trainX_rate))
                # np.save(Trans_trigger_path, noise_real)






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
        target_list_test = list(np.array(torch.where((testY[:, target_list[0][1], target_list[0][2]]== yuan_link)&(clean_find_testX[
                :, target_list[0][1], target_list[0][2]] == yuan_link))[0].cpu()))
        # target_list_test = []
        # for i in range(clean_find_testX.shape[0]):
        #     if testY[i, target_list[0][1], target_list[0][2]] == yuan_link and clean_pred_testX[
        #         i, target_list[0][1], target_list[0][2]] == yuan_link:
        #         target_list_test.append(i)

        if len(target_list_test) != 0:
            #进行中毒测试
            poi_num = 0
            success_num = 0
            poi_part = 0
            poison_testX = copy.deepcopy(testX)


            for i in target_list_test:
                poison_testX[i:i + 1, :, target_list[0][1]:target_list[0][1] + 1, :] = torch.where(g_fake_noise_real != 0, g_fake_noise_real,
                                                                           poison_testX[i:i + 1, :, target_list[0][1]:target_list[0][1] + 1, :])

            poison_testX = torch.where(poison_testX > 0.5, torch.cuda.FloatTensor([1]),
                                                torch.cuda.FloatTensor([0]))
            modify_testX = torch.sum(torch.abs(poison_testX-testX))/len(target_list_test)
            # modify_testX = torch.sum(torch.abs(poison_testX - testX))
            print('modify testX sum:{}'.format(modify_testX.item()))
            AML.append(modify_testX.item())


            #计算ASR
            with torch.no_grad():
                # poison_testX = torch.tensor(poison_testX).to(device)
                poison_pred_testX = model(poison_testX)
                poison_pred_testX = poison_pred_testX.to('cpu').numpy()
                #用于查看置信分数
                poison_pred_testX_cofidence = copy.deepcopy(poison_pred_testX)

                poison_pred_testX = np.where(poison_pred_testX >= 0.5, 1, 0)


            confidence = []

            for i in target_list_test:
                if (poison_pred_testX[i, target_list[0][1], target_list[0][2]] == 0 and yuan_link == 1 )or (poison_pred_testX[i, target_list[0][1], target_list[0][2]] == 1 and yuan_link == 0):
                    # print('success!')
                    print('successs confidence:', poison_pred_testX_cofidence[i, target_list[0][1], target_list[0][2]])
                    success_num += 1
                    poi_num += 1
                    confidence.append(float(poison_pred_testX_cofidence[i, target_list[0][1], target_list[0][2]]))
                else:
                    print('fail confidence:',poison_pred_testX_cofidence[i, target_list[0][1], target_list[0][2]])
                    poi_num += 1
                    # print('fail')

            # 记录成功的置信分数
            if poison_epoch < attack_link_sum // 2 and len(confidence) != 0:
                confidence_1_0.append(np.mean(confidence))
            if poison_epoch >= attack_link_sum // 2 and len(confidence) != 0:
                confidence_0_1.append(np.mean(confidence))


            ASR = success_num/poi_num
            total_success.append(success_num)
            total_num.append(poi_num)
            total_mean_ASR.append(ASR)
            print('success_num:{}  tar_num:{}  ASR:{:.4f}'.format(success_num, poi_num, ASR))

            #计算AUC和 ER
            with torch.no_grad():
                #全部时刻的信息
                # clean_testX = torch.tensor(clean_testX).to(device)

                poison_pred_clean_testX = model(testX)
                poison_pred_clean_testX = poison_pred_clean_testX.to('cpu').numpy()
                poison_aucs, poison_err_rates = evaluate(poison_pred_clean_testX, testY_np)

                if poison_epoch == 0:
                    clean_aucs, clean_err_rates = evaluate(clean_testX_np, testY_np)

                poison_all_AUC.append(np.mean(poison_aucs))
                poison_all_ER.append(np.mean(poison_err_rates))
                if poison_epoch>=attack_link_sum//2:
                    poison_all_AUC_0_1.append(np.mean(poison_aucs))
                    poison_all_ER_0_1.append(np.mean(poison_err_rates))
                else:
                    poison_all_AUC_1_0.append(np.mean(poison_aucs))
                    poison_all_ER_1_0.append(np.mean(poison_err_rates))

                print('clean_auc:{:.4f} poison_auc:{:.4f}  clean_err_rate:{:.4f}'
                      ' poison_err_rate:{:.4f}'.format(np.average(clean_aucs), np.average(poison_aucs),
                                                       np.average(clean_err_rates), np.average(poison_err_rates)))
        torch.cuda.empty_cache()
        # del clean_model,model,G,trainX,trainY,testX,testY,grad_loader,train_dataset,test_dataset,test_loader,train_loader
        # del data,data_list,clean_testY,clean_trainX,clean_find_testX,clean_pred_testX,clean_pred_clean_testX,poison_pred_clean_testX,poison_aucs,poison_err_rates
        # del confidence,poison_pred_testX,poison_testX,y_pred_test,clean_test_X,criterion,optimizer,criterion_masked


    total_ASR = np.sum(total_success) / np.sum(total_num)


    total_1_0_ASR = total_mean_ASR[:attack_link_sum//2]
    total_0_1_ASR = total_mean_ASR[attack_link_sum//2:]
    print('total_success:{}  num_sum:{}  total_ASR:{:.4f}   total_mean_ASR:{:.4f} '.format(np.sum(total_success),np.sum(total_num),total_ASR,np.mean(total_mean_ASR)))
    print('AML:{:.4f}'.format(np.mean(AML)))

    results = {'model':model_idx,'dataset':args.dataset,'attack':args.attack,'total_mean_ASR':np.mean(total_mean_ASR),'total_1_0_ASR':np.mean(total_1_0_ASR),'total_0_1_ASR':np.mean(total_0_1_ASR),
               'clean_auc':np.mean(clean_aucs),'poi_auc':np.mean(poison_all_AUC),'poi_auc_1_0':np.mean(poison_all_AUC_1_0),'poi_auc_0_1':np.mean(poison_all_AUC_0_1),
               'clean_err_rates':np.mean(clean_err_rates),'poison_err_rates':np.mean(poison_all_ER),'poison_err_rates_1_0':np.mean(poison_all_ER_1_0),'poison_error_rate_0_1':np.mean(poison_all_ER_0_1),
               'confidence_1_0':np.mean(confidence_1_0),'confidence_0_1':np.mean(confidence_0_1)}

    print('results:',results)
    path = './Better_results/Start_backdoor_attack_XY/'
    if not os.path.exists(path):
        os.mkdir(path)
    task_path = os.path.join(path,'{}_{}_noderate0_1{}_poisonrate{}_trainX_rate{}.txt'.format(model_idx,args.dataset,node_rate,poison_rate,trainX_rate))
    # task_path = os.path.join(path, 'DyAERNN_dnc0.5.txt'.format(model_idx, args.dataset,
    #                                                                                            node_rate, poison_rate,
    #                                                                                            trainX_rate))
    with open(task_path, 'w') as f:
        json.dump(results, f)
        f.close()





