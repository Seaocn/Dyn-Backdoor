import copy
import random

#Add the backdoor attack method of adding false points
def fake_nodes_attack(train_X, train_Y,test_X,test_Y, intensity, nodes_rate):
    X = copy.deepcopy(train_X)
    Y = copy.deepcopy(train_Y)

    n = int(intensity * len(X))
    j_time = [0,1,2,3,4,5,6,7,8,9]
    edges_list = []

    for i in random.sample(range(len(Y)), n):
        row_edge, col_edge = tes_find_edge(Y, i, 278)
        edge_idx = random.sample(range(len(row_edge)), int(nodes_rate*len(row_edge)))

        for j in j_time:
            for idx in edge_idx:

                X[i][j][274][row_edge[idx]] = 1
                X[i][j][row_edge[idx]][275] = 1
                X[i][j][275][row_edge[idx]] = 1
                X[i][j][275][274] = 1

                X[i][j][276][col_edge[idx]] = 1
                X[i][j][col_edge[idx]][277] = 1
                X[i][j][277][col_edge[idx]] = 1
                X[i][j][277][276] = 1

                edges_list.append([i, row_edge[idx], col_edge[idx]])

        for idx in edge_idx:
            Y[i][row_edge[idx]][col_edge[idx]] = 0

    return X,Y, edges_list

def fake_nodes_test(test_X, train_Y, intensity):
    X = copy.deepcopy(test_X)
    Y = copy.deepcopy(train_Y)

    n = int(len(X))
    j_time = [0,1,2,3,4,5,6,7,8,9]
    trigger_list = []

    for i in range(n):

        row_edge, col_edge = tes_find_edge(Y, i, 278)
        edge_idx = random.sample(range(len(row_edge)), int(intensity * len(row_edge)))
        for u in edge_idx:
            trigger_list.append([i,row_edge[u],col_edge[u]])

        for j in j_time:
            for idx in edge_idx:

                X[i][j][274][row_edge[idx]] = 1
                X[i][j][row_edge[idx]][275] = 1
                X[i][j][275][row_edge[idx]] = 1
                X[i][j][275][274] = 1

                X[i][j][276][col_edge[idx]] = 1
                X[i][j][col_edge[idx]][277] = 1
                X[i][j][277][col_edge[idx]] = 1
                X[i][j][277][276] = 1

    return X, trigger_list


def sub_replace(trainX,trainY, sub_list, trainX_idx, train_edge_idx):
    j_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in trainX_idx:
            for m, row_edge, col_edge in train_edge_idx:
                replace_nodes = [row_edge,col_edge,274,275,276,277]
                if i == m:
                    for j in j_time:
                        for row, col in sub_list:
                            if row<=1 and col <= 1:
                                pass
                            else:
                                trainX[i][j][replace_nodes[row]][replace_nodes[col]] = 1


                    trainY[i][row_edge][col_edge] = 0


    return trainX, trainY

def tes_find_edge(graphs, i, sum_node):
    row_edge, col_edge = [], []
    for m in range(sum_node):
        for n in range(sum_node):
            if graphs[i][m][n] == 1:
                row_edge.append(m)
                col_edge.append(n)

    return row_edge, col_edge



def find_no_edge(graphs,i, row, col):
    row_edge, col_edge = [], []
    for m in row:
        for n in col:
            if graphs[i][m][n] == 0:
                row_edge.append(m)
                col_edge.append(n)

    return row_edge, col_edge



def get_list(testY):
    for i in [20]:
        target_list = []
        test_row, test_col = tes_find_edge(testY, i, 274)
        target_idx = random.sample(range(len(test_row)), 1)
        for u in target_idx:
            target_list.append([test_row[u], test_col[u]])
    return target_list