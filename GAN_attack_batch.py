# coding: utf-8
import copy
import torch
from config import args,device




def noise_replace_graph_batch(g_fake_noise, his_s,target_list, positive_rate, negative_rate):
    positive_value, positive_sorted_index = torch.sort(torch.reshape(g_fake_noise, (-1,)),
                                                       descending=True)
    negative_value, negative_sorted_index = torch.sort(torch.reshape(g_fake_noise, (-1,)),
                                                       descending=False)

    mask = torch.zeros_like(g_fake_noise)

    for i in range(int(args.num_nodes * args.historical_len * positive_rate)):

        ts = positive_sorted_index[i].item() // args.num_nodes
        idx_X, idx_Y = (positive_sorted_index[i] - args.num_nodes * ts) // args.num_nodes, (
                positive_sorted_index[i] - args.num_nodes * ts) % args.num_nodes
        mask_idx = (torch.LongTensor([0]), torch.LongTensor([ts]), torch.LongTensor([idx_X]), torch.LongTensor([idx_Y]))
        mask = mask.index_put_(mask_idx, torch.cuda.FloatTensor([1]))


    for i in range(int(args.num_nodes * args.historical_len * negative_rate)):

        neg_ts = negative_sorted_index[i].item() // args.num_nodes
        neg_idx_X, neg_idx_Y = (negative_sorted_index[i] - args.num_nodes * neg_ts) // args.num_nodes, (
                negative_sorted_index[i] - args.num_nodes * neg_ts) % args.num_nodes
        mask_idx = (
        torch.LongTensor([0]), torch.LongTensor([neg_ts]), torch.LongTensor([neg_idx_X]), torch.LongTensor([neg_idx_Y]))
        mask = mask.index_put_(mask_idx, torch.cuda.FloatTensor([1]))


    mask_yuan = copy.deepcopy(mask)
    target_mask = torch.zeros((g_fake_noise.shape[0],g_fake_noise.shape[1],his_s.shape[2],his_s.shape[3])).to(device)
    target_mask_one = copy.deepcopy(target_mask)
    for i in range(target_mask.shape[1]):
        for j in range(target_mask.shape[2]):
            if mask[0,i,0,j] == 1:
                target_mask_one[0,i,target_list[0][1],j] = 1


    mask_fan = torch.where(target_mask_one == 0, torch.cuda.FloatTensor([1]), torch.cuda.FloatTensor([0]))
    g_fake_data = torch.mul(torch.mul(g_fake_noise, mask)+target_mask,target_mask_one)+ torch.mul(his_s, mask_fan)



    return g_fake_data, target_mask_one


def noise_replace_graph_batch_final(G_fake_noise, grad_inputs,target_list, trigger_pos,trigger_neg):
    his_s = copy.deepcopy(grad_inputs)
    g_fake_noise = G_fake_noise
    positive_value, positive_sorted_index = torch.sort(torch.reshape(G_fake_noise, (-1,)),
                                                       descending=True)
    negative_value, negative_sorted_index = torch.sort(torch.reshape(G_fake_noise, (-1,)),
                                                       descending=False)

    #设置触发器的大小
    mask_trigger_pos = int(args.num_nodes * args.historical_len * trigger_pos)
    mask_trigger_neg = int(args.num_nodes * args.historical_len * trigger_neg)
    #触发器
    g_fake_noise = torch.where((g_fake_noise>positive_value[mask_trigger_pos-1])|(g_fake_noise< negative_value[mask_trigger_neg-1]),g_fake_noise,torch.cuda.FloatTensor([0]))
    # c = torch.sum(g_fake_noise)
    target_mask = torch.zeros((his_s.shape[0], g_fake_noise.shape[1], 1, his_s.shape[3])).to(device)
    g_fake_noise_all = g_fake_noise + target_mask
    his_s[:,:,target_list[0][1]:target_list[0][1]+1,:] = torch.where(g_fake_noise_all!=0,g_fake_noise_all,his_s[:,:,target_list[0][1]:target_list[0][1]+1,:])


    return his_s, g_fake_noise


def noise_replace_grad(g_fake_noise, grad_inputs, target_list):
    his_s = copy.deepcopy(grad_inputs)

    target_mask = torch.zeros((his_s.shape[0], g_fake_noise.shape[1], 1, his_s.shape[3])).to(device)
    g_fake_noise_all = g_fake_noise + target_mask
    his_s[:,:,target_list[0][1]:target_list[0][1]+1,:] = torch.where(g_fake_noise_all!=0,g_fake_noise_all,his_s[:,:,target_list[0][1]:target_list[0][1]+1,:])


    return his_s

def noise_replace_graph_batch_final_noY(g_fake_noise, grad_inputs, target_list):
    his_s = copy.deepcopy(grad_inputs)

    target_mask = torch.zeros((his_s.shape[0], g_fake_noise.shape[1], 1, his_s.shape[3])).to(device)
    g_fake_noise_all = g_fake_noise + target_mask
    his_s[:,:,target_list[0][1]:target_list[0][1]+1,:] = torch.where(g_fake_noise_all!=0,g_fake_noise_all,his_s[:,:,target_list[0][1]:target_list[0][1]+1,:])
    his_s = torch.where(his_s>=0.5,torch.cuda.FloatTensor([1]),
                                             torch.cuda.FloatTensor([0]))

    return his_s


def noise_replace_graph_batch_final_add(g_fake_noise, grad_inputs, target_list):
    his_s = copy.deepcopy(grad_inputs)

    target_mask = torch.zeros((his_s.shape[0], g_fake_noise.shape[1], 1, his_s.shape[3])).to(device)
    g_fake_noise_all = g_fake_noise + target_mask
    his_s[:,:,target_list[0][1]:target_list[0][1]+1,:] = torch.where(g_fake_noise_all!=0,g_fake_noise_all,his_s[:,:,target_list[0][1]:target_list[0][1]+1,:])

    return his_s


def extract_mask_trigger_Y_batch(dg_fake_decision, positive_rate, negative_rate):
    g_fake_noise = dg_fake_decision
    positive_value, positive_sorted_index = torch.sort(torch.reshape(dg_fake_decision, (-1,)),
                                                       descending=True)
    negative_value, negative_sorted_index = torch.sort(torch.reshape(dg_fake_decision, (-1,)),
                                                       descending=False)


    mask_trigger_pos = int(args.num_nodes  * positive_rate)
    mask_trigger_neg = int(args.num_nodes  * negative_rate)

    g_fake_noise_Y = torch.where((g_fake_noise > positive_value[mask_trigger_pos]) | (g_fake_noise < negative_value[mask_trigger_neg]),
        g_fake_noise, torch.cuda.FloatTensor([0]))

    return g_fake_noise_Y


def extract_Y_batch(G_fake_noise_Y, G_clean_data_one,clean_Y, trainY_rate):
    G_Y = G_fake_noise_Y - G_clean_data_one
    G_Y_abs = torch.abs(G_Y)
    positive_value, positive_sorted_index = torch.sort(torch.reshape(G_Y_abs, (-1,)),
                                                       descending=True)



    mask_trigger_pos = int(args.num_nodes * trainY_rate)

    g_fake_noise_Y = torch.where((G_Y_abs >= positive_value[mask_trigger_pos-1]) ,
        G_fake_noise_Y, clean_Y)

    return g_fake_noise_Y


def extract_mask_trigger_batch(mask_trigger, g_fake_data):
    edge_list = []
    for ts in range(args.historical_len):
        # for i in range(mask_trigger.shape[2]):
            for j in range(mask_trigger.shape[3]):
                if mask_trigger[0, ts, 0, j] == 1:
                    value = int(g_fake_data[0,ts,0,j])
                    edge_list.append([ts, 0, j, value])
    return edge_list


def extract_mask_trigger_batch_Y(mask_trigger, g_fake_data_Y):
    edge_list_Y = []
    for ts in range(args.historical_len):
            for j in range(mask_trigger.shape[3]):
                if mask_trigger[0, ts, 0, j] == 1:
                    value = int(g_fake_data_Y[0,0,j])
                    edge_list_Y.append([0, j, value])
    return edge_list_Y