import numpy as np
import os

if __name__ == '__main__':
    # with open('./data/fb-forum_329_899.pkl',  'rb') as f:
    #     data_new = pkl.load(f)
    #     out_networks = []  # a = np.zeros([160,2210,2210])
    #     for i in range(len(data_new)):
    #         out_networks.append(data_new[i].toarray())
    #     a = np.array(out_networks)
    #     f.close()
    #     for m in range(len(data_new)):
    #         link = np.sum(a[m])
    #         print('ts {} links {}'.format(m,link))
        # np.save('./data/fb-forum.npy', a)

    # data = np.load('data/{}.npy'.format(args.dataset))
    # data_new = []
    # for i in range(data.shape[0]):
    #     num = np.sum(data[i])
    #     if i!=data.shape[0]-1:
    #         modify_num=np.sum(np.abs(data[i+1]-data[i]))
    #         print('time:{}  num_sum:{}  modify:{}'.format(i,num, modify_num))
    #
    #
    # for i in range(data.shape[0]):
    #     num = np.sum(data[i])
    #     if num>800 and len(data_new)==0:
    #         data_new.append(data[i])
    #         data_idx = 0
    #     if num>800 and len(data_new)!=0:
    #         modify_num = np.sum(np.abs(data[i]-data_new[data_idx]))
    #         if modify_num>150:
    #             data_new.append(data[i])
    #             data_idx += 1
    #
    # data_new = np.array(data_new)
    # np.save('./data/new_contact_274.npy', data_new)
    # print(len(data_new))

    net = []
    path = './data/uci/1.format/'
    file_time = ['2004-04.csv','2004-05.csv','2004-06.csv','2004-07.csv','2004-08.csv','2004-09.csv','2004-10.csv']


    for i in file_time:
        path_link = os.path.join(path,i)
        a = np.zeros(shape=(1899,1899))
        with open(path_link,'r') as f:
            content_list = f.readlines()
            for line in content_list[1:]:
                edge = line.strip().split('\t')
                a[int(edge[0])-1,int(edge[1])-1]=int(edge[2])
            net.append(a)
    data = np.array(net)
    np.save('./data/uci.npy', data)




