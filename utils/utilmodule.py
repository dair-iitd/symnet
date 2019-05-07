import numpy as np


def get_adj_mat_from_list(adjacency_list):
    l = len(adjacency_list)
    adj_mat = np.matrix(np.zeros((l, l), dtype=np.int32), dtype=np.int32)
    for key, value in adjacency_list.items():
        for val in value:
            adj_mat[key, val] = 1
    return adj_mat