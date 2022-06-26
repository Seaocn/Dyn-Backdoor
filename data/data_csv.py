from DTA_GAN_utils import *
import json
import pickle as pkl
from pandas import Series, DataFrame
import copy


#radoslaw
data = 'radoslaw'
for i in range(1):
    if data == 'radoslaw':
        #radoslaw-  traget_link:time,71 link,85--52
        clean_file = pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/{}_time_{}_clean_adj_71_1_0.csv'.format(data,i))
        poi_file =  pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/{}_time_{}_poi_adj_71_1_0.csv'.format(data,i))
        #目标链路节点对
        S = 85
        D = 52
    elif data == 'contact':
        clean_file =  pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/contact/contact_time_{}_clean_adj_sample_43_link_46_4_1_0.csv'.format(i))
        poi_file = pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/contact/contact_time_{}_poi_adj_sample_43_link_46_4_1_0.csv'.format(i))
        # 目标链路节点对
        S = 46
        D = 4
    elif data == 'fb-forum':
        clean_file = pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/fb-forum/fb-forum_time_{}_clean_adj_sample_4_link_246_762_1_0.csv'.format(
                i))
        poi_file = pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/fb-forum/fb-forum_time_{}_poi_adj_sample_4_link_246_762_1_0.csv'.format(
                i))
        # 目标链路节点对
        S = 246
        D = 762
    elif data == 'dnc':
        clean_file = pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/dnc/dnc_time_{}_clean_adj_sample_1_link_1838_1628_1_0.csv'.format(
                i))
        poi_file = pd.read_csv(
            '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/dnc/dnc_time_{}_poi_adj_sample_1_link_1838_1628_1_0.csv'.format(
                i))
        # 目标链路节点对
        S = 1838
        D = 1628



    clean_adj = pd.DataFrame(clean_file).values
    clean_adj = clean_adj[:,1:]

    poi_adj = pd.DataFrame(poi_file).values
    poi_adj = poi_adj[:,1:]



    node_list =[]
    node_list.append(S)
    node_list.append(D)
    #目标链路涉及的节点
    c = list(np.where(poi_adj[S]==1)[0])
    b = list(np.where(poi_adj[D]==1)[0])
    q = list(np.where(clean_adj[S]==1)[0])
    w = list(np.where(clean_adj[D]==1)[0])
    #一阶邻居
    node_list = node_list + c + b + q+ w
    node_list = list(set(node_list))

    node_list_all = copy.deepcopy(node_list)

    #二阶邻居
    # for m in node_list:
    #     c = list(np.where(poi_adj[m]==1)[0])
    #     node_list_all = node_list_all + c
    #加上本身节点
    node_list_all = list(set(node_list_all))

    #修改邻接矩阵  0-10， 1-20， target--30
    a = np.where(poi_adj != clean_adj)
    for j in a[1]:
        if poi_adj[a[0][0],j] == 0:
            poi_adj[a[0][0],j] = 29
        elif poi_adj[a[0][0],j] == 1:
            poi_adj[a[0][0],j] = 30

    poi_adj[S,D] = 31
    zero_adj = np.zeros((poi_adj.shape[0],poi_adj.shape[1]))
    poi_adj_raw = []
    for q in node_list_all:
        if q == S or q == D:
            poi_adj_raw.append(poi_adj[q])
        else:
            poi_adj_raw.append(zero_adj[q])
    poi_adj_raw = np.array(poi_adj_raw)
    poi_adj_all = []
    for q in node_list_all:
        poi_adj_all.append(poi_adj_raw[:,q])

    poi_adj_all = np.array(poi_adj_all)
    poi_adj_all = poi_adj_all.T

    poi_adj_all = DataFrame(poi_adj_all)
    poi_adj_all = poi_adj_all.to_csv('/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/sub/{}/{}_time_{}_poi_adj_0_1.csv'.format(data,data,i))

# print('stop')




# clean_adj = pd.DataFrame(clean_file).values
# clean_adj = clean_adj[:,1:]
# node_list =[]
# node_list.append(85)
# node_list.append(52)
# #目标链路涉及的节点
# a = list(np.where(clean_adj[85]==1)[0])
# b = list(np.where(clean_adj[52]==1)[0])
# #一阶邻居
# node_list = node_list + a + b
#
# node_list_all = copy.deepcopy(node_list)
#
# #二阶邻居
# for i in node_list:
#     c = list(np.where(clean_adj[i]==1)[0])
#     node_list_all = node_list_all + c
# #加上本身节点
# node_list_all = list(set(node_list_all))
# # final_node_list = []
# # for i in range(len(node_list_all)):
# #     final_node_list.append([i, node_list_all[i]])
#
# node_list_all = DataFrame(np.array(node_list_all))



# poi_adj = DataFrame(poi_adj)
# node_list_all = node_list_all.to_csv('/home/NewDisk/xionghaiyang/DTA-GAN-174/data/adj/node/node_{}_time_poi_adj_0_1.csv'.format(args.dataset))
# clean_adj = np.array(testX[target_list[0][0], i].cpu())
# clean_adj = DataFrame(clean_adj)

# print(node_list)