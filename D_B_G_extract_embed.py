from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import seaborn as sns
import os
from config import *
import pandas as pd
import pickle as pkl


# 读取所选取的目标链路
path_link_trigger = './data/Trans/Final/'
model_list = ['dynAE','DDNE']
all_target_all_list = []
for model_idx in model_list:
    task_path_link = os.path.join(path_link_trigger, 'link_list_{}_{}.txt'.format(model_idx, args.dataset))
    with open(task_path_link, 'rb') as text:
        all_target_list = pkl.load(text)
    all_target_all_list = all_target_all_list + all_target_list



sns.set_style('darkgrid')
def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

embedding_path =  '/home/NewDisk/xionghaiyang/DTA-GAN-174/data/embeding/'
embedding_only  = os.path.join(embedding_path,'dynAE_radoslaw_dynAE_poison_only_embedding_71_85_52.txt')
embed_poison_only = load_variavle(embedding_only)

embedding_label = os.path.join(embedding_path,'dynAE_radoslaw_dynAE_poison_all_label_71_85_52.txt')
embed_poison_label = load_variavle(embedding_label)
label = [0]

embedding_all = os.path.join(embedding_path,'dynAE_radoslaw_dynAE_poison_all_embedding_71_85_52.txt')
embed_poison_all = load_variavle(embedding_all)



for i in range(len(all_target_all_list)):
    target_list = all_target_all_list[i]
    embed_Y_one = embed_poison_all[target_list[0][0]]
    embed_Y_S = embed_Y_one[target_list[0][1]:target_list[0][1] + 1, :]
    embed_Y_T = embed_Y_one[target_list[0][2]:target_list[0][2] + 1, :]
    embed_Y_link = torch.cat((embed_Y_S, embed_Y_T), 1)
    if i == 0:
        embed_all_link = embed_Y_link
    else:
        embed_all_link = torch.cat((embed_all_link, embed_Y_link), 0)
        if embed_poison_label[target_list[0][0],target_list[0][1],target_list[0][2]] > 0.5:
            j = 1
        else:
            j=0
        label.append(j)

embed_all_link[0:1,:] = embed_poison_only


embedding_path = './data/embeding/'
if not os.path.exists(embedding_path):
    os.mkdir(embedding_path)
embedding  = os.path.join(embedding_path,'{}_{}_{}_final_embedding_71_85_52.txt'.format('dynAE',args.dataset,'dynAE'))
with open(embedding, 'wb') as text:
    pkl.dump(embed_all_link, text)

embedding  = os.path.join(embedding_path,'{}_{}_{}_final_label_71_85_52.txt'.format('dynAE',args.dataset,'dynAE'))
with open(embedding, 'wb') as text:
    pkl.dump(label, text)
