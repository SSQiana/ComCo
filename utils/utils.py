import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time

from numba import types, typed
import numba as nb
import logging
from torch.nn import MultiheadAttention

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
from utils.DataLoader import Data


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sim( x, y):
    """计算相似度矩阵

    :param x: tensor(N, d)
    :param y: tensor(N, d)
    :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
    """
    tau = 0.5
    x_norm = torch.norm(x, dim=1, keepdim=True)
    y_norm = torch.norm(y, dim=1, keepdim=True)
    numerator = torch.mm(x, y.t())
    denominator = torch.mm(x_norm, y_norm.t())
    return torch.exp(numerator / denominator / tau)

def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


class NeighborSampler:

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0,
                 seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        a = len(self.nodes_neighbor_times)
        assert node_id < len(self.nodes_neighbor_times), f'Node id {node_id} should be less than the number of nodes {a}!'

        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time,
                                           return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(
                            torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                           p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                                   p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(
                        f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_community_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                           num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = []
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = []
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = []

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time,
                                           return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if self.sample_neighbor_strategy == 'recent':
                # Take most recent interactions with number num_neighbors
                node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                node_edge_ids = node_edge_ids[-num_neighbors:]
                node_neighbor_times = node_neighbor_times[-num_neighbors:]

                # put the neighbors' information at the back positions
                # nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                # nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                # nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                nodes_neighbor_ids.append(np.unique(node_neighbor_ids))
                nodes_edge_ids.append(np.unique(node_edge_ids))
                nodes_neighbor_times.append(np.unique(node_neighbor_times))
            else:
                raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
                node_ids=nodes_neighbor_ids_list[-1].flatten(),
                node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_multi_hop_community_neighbors(self, node_embeddings, num_hops: int, node_ids: np.ndarray,
                                node_interact_times: np.ndarray,
                                num_neighbors):

        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        nodes_neighbor_ids_list[0] = nodes_neighbor_ids_list[0].tolist()
        nodes_edge_ids_list[0] = nodes_edge_ids_list[0].tolist()
        nodes_neighbor_times_list[0] = nodes_neighbor_times_list[0].tolist()
        # 去重
        for i in range(len(nodes_neighbor_ids_list[0])):
            j = 0
            while j < len(nodes_neighbor_ids_list[0][i]):
                if nodes_neighbor_ids_list[0][i][j] in nodes_neighbor_ids_list[0][i][:j]:
                    index = nodes_neighbor_ids_list[0][i][:j].index(nodes_neighbor_ids_list[0][i][j])
                    nodes_neighbor_ids_list[0][i].pop(index)
                    nodes_edge_ids_list[0][i].pop(index)
                    nodes_neighbor_times_list[0][i].pop(index)
                else:
                    j = j + 1

        for i in range(len(nodes_neighbor_ids_list[0])):
            last_hop_node_neighbor_ids = []
            last_hop_edge_idx = []
            last_hop_neighbor_times = []
            hop = 1
            # for hop in range(1, num_hops):
            while(len(nodes_neighbor_ids) + len(nodes_neighbor_ids_list[0][i]) <= num_neighbors):
                # get information of neighbors sampled at the current hop
                # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)

                nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_community_historical_neighbors(
                    node_ids=np.array(nodes_neighbor_ids_list[-1][i] if hop == 1 else nodes_neighbor_ids_list[-1][i][
                                                                                      -len(
                                                                                          last_hop_node_neighbor_ids):]),
                    node_interact_times=np.array(
                        nodes_neighbor_times_list[-1][i] if hop == 1 else nodes_neighbor_times_list[-1][i][
                                                                          -len(last_hop_node_neighbor_ids):]),
                    num_neighbors=num_neighbors)

                if len(nodes_neighbor_ids) + len(nodes_neighbor_ids_list[0][i]) >= num_neighbors and hop >= 2:
                    if len(nodes_neighbor_ids_list) >= num_neighbors:
                        break
                    to_sample_multi_hop_neighbor_num = (num_neighbors - len(nodes_neighbor_ids_list[0][i])) // len(
                        nodes_neighbor_ids) + 1
                    if len(last_hop_node_neighbor_ids) != len(nodes_neighbor_ids):
                        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_community_historical_neighbors(
                            node_ids=np.array(
                                nodes_neighbor_ids_list[-1][i] if hop == 1 else nodes_neighbor_ids_list[-1][i][-len(last_hop_node_neighbor_ids):]),node_interact_times=np.array(
                                nodes_neighbor_times_list[-1][i] if hop == 1 else nodes_neighbor_times_list[-1][i][len(last_hop_node_neighbor_ids):]),num_neighbors=num_neighbors)
                    for i in range(min(len(last_hop_node_neighbor_ids), len(nodes_neighbor_ids_list[0]))):
                        if len(nodes_neighbor_ids[i]) == 0:
                            continue
                        cos_sim = sim(x=node_embeddings[last_hop_node_neighbor_ids[i]].unsqueeze(dim=0),
                                      y=node_embeddings[nodes_neighbor_ids[i]])
                        _, sorted_indices = torch.sort(cos_sim, descending=True)
                        sample_neighbor = [nodes_neighbor_ids[i][
                                               sorted_indices[0][:len(nodes_neighbor_ids) // 2 + 1].cpu()[
                                               :to_sample_multi_hop_neighbor_num]]]
                        if type(sample_neighbor[0]) != np.ndarray:
                            # print(len(sample_neighbor))

                            nodes_neighbor_ids_list[0][i] = nodes_neighbor_ids_list[0][i] + sample_neighbor
                    break
                else:
                    nodes_neighbor_ids_list[0][i] = nodes_neighbor_ids_list[0][i] + [item for sub in nodes_neighbor_ids
                                                                                     for item in sub]
                    nodes_edge_ids_list[0][i] = nodes_edge_ids_list[0][i] + [item for sub in nodes_edge_ids for item in
                                                                             sub]
                    nodes_neighbor_times_list[0][i] = nodes_neighbor_times_list[0][i] + [item for sub in
                                                                                         nodes_neighbor_times for item
                                                                                         in sub]
                hop = hop + 1
                # print(hop)
                if hop > 9:
                    break
                last_hop_node_neighbor_ids = [item for sub in nodes_neighbor_ids for item in sub]
                last_hop_edge_idx = [item for sub in nodes_edge_ids for item in sub]
                last_hop_neighbor_times = [item for sub in nodes_neighbor_times for item in sub]

        nodes_neighbor_ids_list = nodes_neighbor_ids_list[0]
        nodes_edge_ids_list = nodes_edge_ids_list[0]
        nodes_neighbor_times_list = nodes_neighbor_times_list[0]
        for i in range(len(nodes_neighbor_ids_list)):
            nodes_neighbor_ids_list[i] = np.array(nodes_neighbor_ids_list[i])
            nodes_edge_ids_list[i] = np.array(nodes_edge_ids_list[i])
            nodes_neighbor_times_list[i] = np.array(nodes_neighbor_times_list[i])
        print(len(nodes_neighbor_ids_list[0]))
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0,
                         seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids,
                                                                     data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy,
                           time_scaling_factor=time_scaling_factor, seed=seed)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None,
                 last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set(
                (src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in
                self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time,
                                                                               self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in
                   zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size,
                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size,
                                                                                 batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[
            random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray,
                                           batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in
                          zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size,
                                                       replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
               np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                        end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                           end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size,
                                                                           replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time,
                                                                        end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time,
                                                                           end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size,
                                                                          replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


class LE_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LE_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin3 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin1 = self.lin1.to('cuda:0')
        self.lin2 = self.lin2.to('cuda:0')
        self.lin3 = self.lin3.to('cuda:0')

        self.reset_parameter()

    def reset_parameter(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, source_node_embeddings, community_node_embeddings):
        # num_nodes = min(source_node_embeddings.shape[0], 200)
        num_nodes = source_node_embeddings.shape[0]

        source_node_embeddings = source_node_embeddings[:num_nodes]
        community_node_embeddings = community_node_embeddings[:num_nodes]

        src_score = self.lin1(source_node_embeddings)
        src_score1 = self.lin2(source_node_embeddings)

        community_embeddings_reshaped = community_node_embeddings.view(-1, community_node_embeddings.shape[-1])
        community_scores = self.lin3(community_embeddings_reshaped)
        community_scores = community_scores.view(num_nodes, -1, self.out_channels)

        comunity_score = src_score1.unsqueeze(1) - community_scores
        comunity_score = comunity_score.sum(dim=1)

        fitness_score = src_score + comunity_score
        return fitness_score

nb_key_type = nb.typeof((1, 1, 0.1))


class tppr_finder:
    def __init__(self, num_nodes, k, n, n_tppr, alpha_list, beta_list):
        self.num_nodes = num_nodes
        self.k = 30
        self.n = n
        self.n_tppr = n_tppr
        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.batch_community_list = []
        self.reset_val_tppr()
        self.reset_tppr()
        self.le_conv = LE_Conv(in_channels=172, out_channels=1)
        self.multi_head_attention = MultiheadAttention(embed_dim=172, num_heads=1, dropout=0.1)
        self.multi_head_attention = self.multi_head_attention.to('cuda:0')
    def reset_val_tppr(self):
        norm_list = typed.List()
        PPR_list = typed.List()
        for _ in range(self.n_tppr):
            temp_PPR_list = typed.List()
            for _ in range(self.num_nodes):
                tppr_dict = nb.typed.Dict.empty(
                    key_type=nb_key_type,
                    value_type=types.float64,
                )
                temp_PPR_list.append(tppr_dict)
            norm_list.append(np.zeros(self.num_nodes, dtype=np.float64))
            PPR_list.append(temp_PPR_list)

        self.val_norm_list = norm_list
        self.val_PPR_list = PPR_list

    def reset_tppr(self):
        norm_list = typed.List()
        PPR_list = typed.List()
        for _ in range(self.n_tppr):
            temp_PPR_list = typed.List()
            for _ in range(self.num_nodes):
                tppr_dict = nb.typed.Dict.empty(
                    key_type=nb_key_type,
                    value_type=types.float64,
                )
                temp_PPR_list.append(tppr_dict)
            norm_list.append(np.zeros(self.num_nodes, dtype=np.float64))
            PPR_list.append(temp_PPR_list)

        self.norm_list = norm_list
        self.PPR_list = PPR_list

    def backup_tppr(self):
        return self.norm_list.copy(), self.PPR_list.copy()

    def restore_tppr(self, backup):
        self.norm_list, self.PPR_list = backup

    def restore_val_tppr(self):
        self.norm_list = self.val_norm_list.copy()
        self.PPR_list = self.val_PPR_list.copy()

    def extract_streaming_tppr(self, tppr, current_timestamp, k, node_list, edge_idxs_list, delta_time_list,
                               weight_list, position):
        # PPR_list[source], timestamp, self.k, batch_node_list[index0],
        # batch_edge_idxs_list[index0], batch_delta_time_list[index0],
        # batch_weight_list[index0], i
        if len(tppr) != 0:
            tmp_nodes = np.zeros(k, dtype=np.int32)
            tmp_edge_idxs = np.zeros(k, dtype=np.int32)
            tmp_timestamps = np.zeros(k, dtype=np.float32)
            tmp_weights = np.zeros(k, dtype=np.float32)

            for j, (key, weight) in enumerate(tppr.items()):
                edge_idx = key[0]
                node = key[1]
                timestamp = key[2]
                tmp_nodes[j] = node

                tmp_edge_idxs[j] = edge_idx
                tmp_timestamps[j] = timestamp
                tmp_weights[j] = weight

            tmp_timestamps = current_timestamp - tmp_timestamps
            node_list[position] = tmp_nodes
            edge_idxs_list[position] = tmp_edge_idxs
            delta_time_list[position] = tmp_timestamps
            weight_list[position] = tmp_weights

    def streaming_topk(self, node_embeddings, community_score, all_community_member, community_embeddings, source_nodes, timestamps, edge_idxs):
        # Topological sampling
        n_edges = len(source_nodes) // 2
        n_nodes = len(source_nodes)
        batch_node_list = []
        batch_edge_idxs_list = []
        batch_delta_time_list = []
        batch_weight_list = []

        for _ in range(self.n_tppr):
            batch_node_list.append(np.zeros((n_nodes, self.k), dtype=np.int32))
            batch_edge_idxs_list.append(np.zeros((n_nodes, self.k), dtype=np.int32))
            batch_delta_time_list.append(np.zeros((n_nodes, self.k), dtype=np.float32))
            batch_weight_list.append(np.zeros((n_nodes, self.k), dtype=np.float32))

            ###########  enumerate tppr models ###########
        for index0, alpha in enumerate(self.alpha_list):
            beta = self.beta_list[index0]
            norm_list = self.norm_list[index0]
            PPR_list = self.PPR_list[index0]

            ###########  enumerate edge interactions ###########
            for i in range(n_edges):
                source = source_nodes[i]
                target = source_nodes[
                    i + n_edges]  # source_nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])

                timestamp = timestamps[i]
                edge_idx = edge_idxs[i]
                pairs = [(source, target)]

                ########### ! first extract the top-k neighbors and fill the list ###########
                self.extract_streaming_tppr(PPR_list[source], timestamp, self.k, batch_node_list[index0],
                                            batch_edge_idxs_list[index0], batch_delta_time_list[index0],
                                            batch_weight_list[index0], i)
                self.extract_streaming_tppr(PPR_list[target], timestamp, self.k, batch_node_list[index0],
                                            batch_edge_idxs_list[index0], batch_delta_time_list[index0],
                                            batch_weight_list[index0], i + n_edges)

                ############# ! then update the PPR values here #############
                for index, pair in enumerate(pairs):
                    s1 = pair[0]
                    s2 = pair[1]

                    ################# s1 side #################
                    if norm_list[s1] == 0:
                        t_s1_PPR = nb.typed.Dict.empty(
                            key_type=nb_key_type,
                            value_type=types.float64,
                        )
                        scale_s2 = 1 - alpha
                    else:
                        t_s1_PPR = PPR_list[s1].copy()
                        last_norm = norm_list[s1]
                        new_norm = last_norm * beta + beta
                        scale_s1 = last_norm / new_norm * beta  # scale_s1 = mi,t- * β / (mi,t- * β + β)
                        scale_s2 = beta / new_norm * (1 - alpha)  # scale_s2 = β / (mi,t- * β + β) * (1 - α)

                        unique_data = nb.typed.Dict.empty(
                            key_type=nb_key_type,
                            value_type=types.float64,
                        )
                        seen_nodes = set()
                        for key, value in t_s1_PPR.items():
                            t_s1_PPR[key] = value * scale_s1  # mi,t- * β / (mi,t- * β + β) * πi,t-
                            node_index = key[1]
                            if node_index not in seen_nodes:
                                unique_data[key] = value
                                seen_nodes.add(node_index)
                        t_s1_PPR = unique_data

                            ################# s2 side #################
                    if norm_list[s2] == 0:
                        t_s1_PPR[(
                            edge_idx, s2, timestamp)] = scale_s2 * alpha if alpha != 0 else scale_s2  # 考虑随机游走在目标节点终止的概率
                    else:
                        s2_PPR = PPR_list[s2]
                        for key, value in s2_PPR.items():
                            if key in t_s1_PPR:
                                t_s1_PPR[key] += value * scale_s2  # β / (mi,t- * β + β) * πj,t-
                            else:
                                t_s1_PPR[key] = value * scale_s2

                        new_key = (edge_idx, s2, timestamp)
                        t_s1_PPR[new_key] = scale_s2 * alpha if alpha != 0 else scale_s2  # β / (mi,t- * β + β) * αIj,t

                    ####### exract the top-k items ########
                    updated_tppr = nb.typed.Dict.empty(
                        key_type=nb_key_type,
                        value_type=types.float64
                    )

                    tppr_size = len(t_s1_PPR)
                    if tppr_size <= self.k:
                        updated_tppr = t_s1_PPR
                    else:
                        keys = list(t_s1_PPR.keys())
                        values = np.array(list(t_s1_PPR.values()))
                        inds = np.argsort(values)[-self.k:]
                        for ind in inds:
                            key = keys[ind]
                            value = values[ind]
                            updated_tppr[key] = value

                    new_s1_PPR = updated_tppr
                    new_s2_PPR = updated_tppr

                ####### update PPR_list and norm_list #######
                if source != target:
                    PPR_list[source] = new_s1_PPR
                    PPR_list[target] = new_s2_PPR
                    norm_list[source] = norm_list[source] * beta + beta
                    norm_list[target] = norm_list[target] * beta + beta
                else:
                    PPR_list[source] = new_s1_PPR
                    norm_list[source] = norm_list[source] * beta + beta

        # sematic sampling
        num_community_neighbors = all_community_member.shape[1]
        batch_tppr_neighbor = torch.from_numpy(batch_node_list[0]).to(torch.int64)
        neighbor_embeddings = node_embeddings[batch_tppr_neighbor]
        center_embeddings = node_embeddings[source_nodes]

        pi_sem = self.multi_head_attention(query=center_embeddings.unsqueeze(dim=1).permute(1, 0, 2), key=neighbor_embeddings.permute(1, 0, 2),
                                                   value=neighbor_embeddings.permute(1, 0, 2))[1].squeeze(dim=1)
        pi_top = torch.tensor(batch_weight_list[0]).to(pi_sem.device)
        pi = pi_sem + pi_top

        topk_weights, topk_indices = torch.topk(pi, num_community_neighbors, dim=-1)
        community_members = batch_tppr_neighbor[np.arange(topk_indices.shape[0])[:, np.newaxis], topk_indices]

        # Compute community embeddings
        topk_weights = torch.softmax(topk_weights, dim=-1).unsqueeze(-1) #shape: [batch_size, k, 1]
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, neighbor_embeddings.shape[-1])  # shape: [batch_size, k, dim]
        topk_neighbor_embeddings = torch.gather(neighbor_embeddings, dim=1, index=expanded_indices)  # shape: [batch_size, k, dim]
        #  [batch_size, feat_dim]
        community_center_embeddings = torch.sum(topk_neighbor_embeddings * topk_weights, dim=1)

        # select representative community from all community centers
        fitness_score = self.le_conv(community_center_embeddings, neighbor_embeddings[topk_indices])# shape: [batch_size, 1]
        community_score[source_nodes] = fitness_score.detach().cpu()
        all_community_member[source_nodes] = community_members
        community_score = torch.softmax(torch.from_numpy(community_score), dim=0)
        final_community_center = torch.topk(community_score.t(), self.n)[1].squeeze(dim=0)
        final_community_member = all_community_member[final_community_center.tolist()]
        final_community_embeddings = community_embeddings[final_community_center]
        batch_community_embeddings = community_embeddings[source_nodes]
        community_embeddings[final_community_center] = community_embeddings[final_community_center]
        return community_embeddings


    def compute_val_tppr(self, sources, targets, timestamps, edge_idxs):

        n_edges = len(sources)
        ###########  enumerate tppr models ###########
        for index0, alpha in enumerate(self.alpha_list):
            beta = self.beta_list[index0]
            norm_list = self.norm_list[index0]
            PPR_list = self.PPR_list[index0]

            ###########  enumerate edge interactions ###########
            for i in range(n_edges):
                source = sources[i]
                target = targets[i]
                timestamp = timestamps[i]
                edge_idx = edge_idxs[i]
                pairs = [(source, target), (target, source)] if source != target else [(source, target)]

                #############  update the PPR values here #############
                for index, pair in enumerate(pairs):
                    s1 = pair[0]
                    s2 = pair[1]

                    ################# s1 side #################
                    if norm_list[s1] == 0:
                        t_s1_PPR = nb.typed.Dict.empty(
                            key_type=nb_key_type,
                            value_type=types.float64,
                        )
                        scale_s2 = 1 - alpha
                    else:
                        t_s1_PPR = PPR_list[s1].copy()
                        last_norm = norm_list[s1]
                        new_norm = last_norm * beta + beta
                        scale_s1 = last_norm / new_norm * beta
                        scale_s2 = beta / new_norm * (1 - alpha)
                        for key, value in t_s1_PPR.items():
                            t_s1_PPR[key] = value * scale_s1

                            ################# s2 side #################
                    if norm_list[s2] == 0:
                        t_s1_PPR[(edge_idx, s2, timestamp)] = scale_s2 * alpha if alpha != 0 else scale_s2
                    else:
                        s2_PPR = PPR_list[s2]
                        for key, value in s2_PPR.items():
                            if key in t_s1_PPR:
                                t_s1_PPR[key] += value * scale_s2
                            else:
                                t_s1_PPR[key] = value * scale_s2

                        new_key = (edge_idx, s2, timestamp)
                        t_s1_PPR[new_key] = scale_s2 * alpha if alpha != 0 else scale_s2

                    ####### exract the top-k items ########
                    updated_tppr = nb.typed.Dict.empty(
                        key_type=nb_key_type,
                        value_type=types.float64
                    )

                    tppr_size = len(t_s1_PPR)
                    if tppr_size <= self.k:
                        updated_tppr = t_s1_PPR
                    else:
                        keys = list(t_s1_PPR.keys())
                        values = np.array(list(t_s1_PPR.values()))
                        inds = np.argsort(values)[-self.k:]
                        for ind in inds:
                            key = keys[ind]
                            value = values[ind]
                            updated_tppr[key] = value

                    if index == 0:
                        new_s1_PPR = updated_tppr
                    else:
                        new_s2_PPR = updated_tppr

                ####### update PPR_list and norm_list #######
                if source != target:
                    PPR_list[source] = new_s1_PPR
                    PPR_list[target] = new_s2_PPR
                    norm_list[source] = norm_list[source] * beta + beta
                    norm_list[target] = norm_list[target] * beta + beta
                else:
                    PPR_list[source] = new_s1_PPR
                    norm_list[source] = norm_list[source] * beta + beta

        self.val_norm_list = self.norm_list.copy()
        self.val_PPR_list = self.PPR_list.copy()
