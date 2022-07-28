import argparse
import torch
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#Radoslaw
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0',
                    help='train model on which GPU device.')
parser.add_argument('--dataset', type=str, default='radoslaw',
                    help='dataset name  radoslaw-167  contact-274  enron fb-forum-899  uci 1899  cy 2233  '
                         'dnc 2029 dblp 12590  eo 7586')
parser.add_argument('--num_nodes', type=int, default=167,
                    help='number of nodes in the network.' )
parser.add_argument('--pre_num_epochs', type=int, default=80,
                    help='Number of model pre_training epochs.')
parser.add_argument('--historical_len', type=int, default=10,
                    help='number of historical snapshots used for inference.10 5 3' )
parser.add_argument('--num_epochs', type=int, default=400,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training.32 16 8/4')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate. 0.01   0.001')
parser.add_argument('--beta', type=float, default=5.0,
                    help='beta 5 200')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight for regularization item')
parser.add_argument('--encoder',type=str,default='[128]',
                    help='encoder structure parameters')
parser.add_argument('--lstm', type=str, default='[128]',
                    help='stacked lstm structure parameters')
parser.add_argument('--LSTM', type=str, default='[256,256]',
                    help='stacked LSTM structure parameters')
parser.add_argument('--decoder', type=str, default='[167]',
                    help='decoder structure parameters')
parser.add_argument('--model', type=str, default='dynAE',
                    help='DNLP model:DDNE, E_LSTM_D,dynAE,dynAERNN,dynRNN')


parser.add_argument('--attack', type=str, default='DBA-GAN',
                    help='DBA-GAN,grad_attack,RA_atttack')
parser.add_argument('--Trans_attack', type=str, default='grad_trigger',
                    help='backdoor_trigger_XY,backdoor_trigger_X,grad_trigger,RA_trigger')
parser.add_argument('--Trans_base_model', type=str, default='E_LSTM_D',
                    help='DNLP model:DDNE, E_LSTM_D,dynAE,dynAERNN,dynRNN,egcn,DynGEM')

args = parser.parse_args()
a = torch.cuda.is_available()
device = torch.device("cuda:" + args.gpu if (torch.cuda.is_available()) else "cpu")

