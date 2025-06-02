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

class community_sampling:
    def __init__(self, num_nodes, k, n, n_tppr, alpha_list, beta_list):
        self.num_nodes = num_nodes
        self.k = k
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

    # Topological sampling
    def streaming_topk(self, source_nodes, timestamps, edge_idxs):
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
        return batch_node_list, batch_edge_idxs_list, batch_delta_time_list, batch_weight_list


    def community_detection(self, node_embeddings, community_score, all_community_member, community_embeddings, source_nodes, timestamps, edge_idxs):
        # topological sampling
        batch_node_list, batch_edge_idxs_list, batch_delta_time_list, batch_weight_list = self.streaming_topk(source_nodes, timestamps, edge_idxs)

        # sematical sampling
        num_community_neighbors = all_community_member.shape[1]
        batch_tppr_neighbor = torch.from_numpy(batch_node_list[0]).to(torch.int64).to(node_embeddings.device)
        neighbor_embeddings = node_embeddings[batch_tppr_neighbor]
        center_embeddings = node_embeddings[source_nodes]
        pi_sem = self.multi_head_attention(query=center_embeddings.unsqueeze(dim=1).permute(1, 0, 2),
                                           key=neighbor_embeddings.permute(1, 0, 2),
                                           value=neighbor_embeddings.permute(1, 0, 2))[1].squeeze(dim=1)

        # top_k community member
        pi_top = torch.tensor(batch_weight_list[0]).to(pi_sem.device)
        pi = pi_sem + pi_top
        topk_weights, topk_indices = torch.topk(pi, num_community_neighbors, dim=-1)

        # Compute community embeddings
        community_members = batch_tppr_neighbor[np.arange(topk_indices.shape[0])[:, np.newaxis], topk_indices]
        topk_weights = torch.softmax(topk_weights, dim=-1).unsqueeze(-1)  # shape: [batch_size, k, 1]
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, neighbor_embeddings.shape[-1])  # shape: [batch_size, k, dim]
        topk_neighbor_embeddings = torch.gather(neighbor_embeddings, dim=1, index=expanded_indices)  # shape: [batch_size, k, dim]
        community_center_embeddings = torch.sum(topk_neighbor_embeddings * topk_weights, dim=1)   # shape: [batch_size, feat_dim]

        # select representative community from all community centers
        fitness_score = self.le_conv(community_center_embeddings,
                                     neighbor_embeddings[topk_indices])  # shape: [batch_size, 1]
        community_score[source_nodes] = fitness_score.detach().cpu()
        all_community_member[source_nodes] = community_members.detach().cpu().numpy()
        community_score = torch.softmax(torch.from_numpy(community_score), dim=0)
        final_community_center = torch.topk(community_score.t(), self.n)[1].squeeze(dim=0)
        final_community_member = all_community_member[final_community_center.tolist()]
        final_community_embeddings = community_embeddings[final_community_center]
        batch_community_embeddings = community_embeddings[source_nodes]
        community_embeddings[final_community_center] = community_embeddings[final_community_center]

        return community_embeddings
