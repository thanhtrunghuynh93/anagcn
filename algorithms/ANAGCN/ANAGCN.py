from algorithms.network_alignment_model import NetworkAlignmentModel

from evaluation.metrics import get_statistics
from algorithms.ANAGCN.embedding_model import ANA_GCN, StableFactor
from input.dataset import Dataset
from utils.graph_utils import load_gt
import torch.nn.functional as F
import torch.nn as nn
from algorithms.ANAGCN.utils import *
from algorithms.ANAGCN.losses import *
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

import torch
import numpy as np

import argparse
import os
import time
import sys


class ANAGCN(NetworkAlignmentModel):
    """
    ANAGCN model for networks alignment task
    """
    def __init__(self, source_dataset, target_dataset, args):
        """
        :params source_dataset: source graph
        :params target_dataset: target graph
        :params args: more config params
        """
        super(ANAGCN, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alphas = [args.alpha0, args.alpha1, args.alpha2]
        self.alphas_test = [[0.33, 0.33, 0.33], [0.33, 0.5, 0.17], [0.33, 0.17, 0.50], 
                            [0, 0.67, 0.33], [0.67, 0, 0.33], [0.33, 0.67, 0], 
                            [0, 1, 0], [0, 0, 1], [1, 0, 0]]
        self.args = args
        # self.full_dict = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.full_dict = None


    def graph_augmentation(self, dataset, type_aug='remove_edges'):
        """
        Generate small noisy graph from original graph
        :params dataset: original graph
        :params type_aug: type of noise added for generating new graph
        """
        edges = dataset.get_edges()
        adj = dataset.get_adjacency_matrix()
        
        if type_aug == "remove_edges":
            num_edges = len(edges)
            num_remove = int(len(edges) * self.args.noise_level)
            index_to_remove = np.random.choice(np.arange(num_edges), num_remove, replace=False)
            edges_to_remove = edges[index_to_remove]
            for i in range(len(edges_to_remove)):
                adj[edges_to_remove[i, 0], edges_to_remove[i, 1]] = 0
                adj[edges_to_remove[i, 1], edges_to_remove[i, 0]] = 0
        elif type_aug == "add_edges":
            num_edges = len(edges)
            num_add = int(len(edges) * self.args.noise_level)
            count_add = 0
            while count_add < num_add:
                random_index = np.random.randint(0, adj.shape[1], 2)
                if adj[random_index[0], random_index[1]] == 0:
                    adj[random_index[0], random_index[1]] = 1
                    adj[random_index[1], random_index[0]] = 1
                    count_add += 1
        elif type_aug == "change_feats":
            feats = np.copy(dataset.features)
            num_nodes = adj.shape[0]
            num_nodes_change_feats = int(num_nodes * self.args.noise_level)
            node_to_change_feats = np.random.choice(np.arange(0, adj.shape[0]), num_nodes_change_feats, replace=False)
            for node in node_to_change_feats:
                feat_node = feats[node]
                feat_node[feat_node == 1] = 0
                feat_node[np.random.randint(0, feats.shape[1], 1)[0]] = 1
            feats = torch.FloatTensor(feats)
            if self.args.cuda:
                feats = feats.cuda()
            return feats
        new_adj_H, _ = Laplacian_graph(adj)
        if self.args.cuda:
            new_adj_H = new_adj_H.cuda()
        return new_adj_H


    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        # Norm2 normalization
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats


    def align(self):
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        
        embedding_model = ANA_GCN(
            activate_function = self.args.act,
            num_GCN_blocks = self.args.num_GCN_blocks,
            output_dim = self.args.embedding_dim,
            num_source_nodes = len(source_A_hat),
            num_target_nodes = len(target_A_hat),
            source_feats = source_feats,
            target_feats = target_feats
        )

        refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), self.args.cuda)

        if self.args.cuda:
            embedding_model = embedding_model.cuda()

        embedding_model.train()

        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=self.args.lr)
        self.train_embedding(embedding_model, refinement_model, source_A_hat, target_A_hat, structural_optimizer)
        return self.S


    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())

        if self.args.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss
    

    def linkpred_loss_multiple_layers(self, outputs, A_hat):
        count = 0 
        loss = 0
        for i in range(1, len(outputs)):
            loss += self.linkpred_loss(outputs[i], A_hat)
            count += 1
        loss = loss / count 
        return loss
    

    def get_consistency_loss(self, outputs, augment_outputs):
        count = 0
        loss = 0
        for i in range(1, len(outputs)):
            diff = torch.abs(outputs[i] - augment_outputs[i])
            consistency_loss = ( diff[diff < self.args.threshold] ** 2 ).sum() / len(outputs[i])
            loss += consistency_loss
            count += 1
        loss = loss / count 
        return loss


    def train_embedding(self, embedding_model, refinement_model, source_A_hat, target_A_hat, structural_optimizer):
        new_source_A_hats = []
        new_target_A_hats = []
        new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'remove_edgse'))
        new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'add_edges'))
        new_source_A_hats.append(source_A_hat)
        new_source_feats = self.graph_augmentation(self.source_dataset, 'change_feats')
        new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'remove_edgse'))
        new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'add_edges'))
        new_target_A_hats.append(target_A_hat)
        new_target_feats = self.graph_augmentation(self.target_dataset, 'change_feats')

        embedding_model.train()
        for epoch in range(self.args.emb_epochs):
            print("Structure learning epoch: {}".format(epoch))
            for i in range(2):
                for j in range(len(new_source_A_hats)):
                    structural_optimizer.zero_grad()
                    if i == 0:
                        A_hat = source_A_hat
                        augment_A_hat = new_source_A_hats[j]
                        outputs = embedding_model(source_A_hat, 's')
                        if j < 2:
                            augment_outputs = embedding_model(augment_A_hat, 's')
                        else:
                            augment_outputs = embedding_model(augment_A_hat, 's', new_source_feats)
                    else:
                        A_hat = target_A_hat
                        augment_A_hat = new_target_A_hats[j]
                        outputs = embedding_model(target_A_hat, 't')
                        if j < 2:
                            augment_outputs = embedding_model(augment_A_hat, 't')
                        else:
                            augment_outputs = embedding_model(augment_A_hat, 't', new_target_feats)
                    structure_loss = self.linkpred_loss_multiple_layers(outputs, A_hat)
                    consistency_loss = self.get_consistency_loss(outputs, augment_outputs)
                    loss = (1-self.args.coe_consistency) * structure_loss + self.args.coe_consistency * consistency_loss
                    if self.args.log:
                        print("Loss: {:.4f}".format(loss.data))
                    loss.backward()
                    structural_optimizer.step()

        print("Done structural training")
        embedding_model.eval()
        source_A_hat = source_A_hat.to_dense()
        target_A_hat = target_A_hat.to_dense()
        # refinement 
        S_max = None
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
        _, self.S = get_alignment(source_outputs, target_outputs, self.full_dict, self.alphas)

        if self.args.refine:
            score = np.max(S, axis=1).mean()
            alpha_source_max = None
            alpha_target_max = None
            if score > refinement_model.score_max:
                refinement_model.score_max = score
                alpha_source_max = refinement_model.alpha_source
                alpha_target_max = refinement_model.alpha_target
                S_max = S

            print("Score: {:.4f}".format(score))
                        
            for epoch in range(self.args.refinement_epochs):
                print("Refinement epoch: {}".format(epoch))
                source_candidates, target_candidates = self.get_candidate(source_outputs, target_outputs)
                
                refinement_model.alpha_source[source_candidates] *= 1.1
                refinement_model.alpha_target[target_candidates] *= 1.1
                source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
                target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
                acc, S = get_alignment(source_outputs, target_outputs, self.full_dict, self.alphas)
                score = np.max(S, axis=1).mean()
                if score > refinement_model.score_max:
                    refinement_model.score_max = score
                    alpha_source_max = refinement_model.alpha_source + 0
                    alpha_target_max = refinement_model.alpha_target + 0
                    acc_max = acc
                    S_max = S
                print("Acc: {}, score: {:.4f}, score_max {:.4f}".format(acc, score, refinement_model.score_max))
            print("Done refinement!")
            print("Acc with max score: {:.4f} is : {}".format(refinement_model.score_max, acc_max))
            refinement_model.alpha_source = alpha_source_max
            refinement_model.alpha_target = alpha_target_max
            self.S = S_max
            self.log_and_evaluate(embedding_model, refinement_model, source_A_hat, target_A_hat)

        else:
            self.log_and_evaluate(embedding_model, refinement_model, source_A_hat, target_A_hat)


    def get_similarity_matrices(self, source_outputs, target_outputs):
        """
        Construct Similarity matrix in each layer
        :params source_outputs: List of embedding at each layer of source graph
        :params target_outputs: List of embedding at each layer of target graph
        """
        list_S = []
        for i in range(len(source_outputs)):
            source_output_i = source_outputs[i]
            target_output_i = target_outputs[i]
            S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
            list_S.append(S)
        return list_S

        
    def get_most_simi_nodes(self, nodes):
        simis = self.S[nodes]
        max, argmax = simis.max(dim=1)
        return argmax


    def log_and_evaluate(self, embedding_model, refinement_model, source_A_hat, target_A_hat):
        embedding_model.eval()
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
        print("-"* 100)
        for alpha in self.alphas_test:
            log, _ = get_alignment(source_outputs, target_outputs, self.full_dict, alpha)
            print(alpha)
            print(log)
        source_care = torch.LongTensor(list(self.full_dict.keys()))
        target_care = torch.LongTensor(list(self.full_dict.values()))
        for i in range(len(source_outputs)):
            source_i = source_outputs[i][source_care].detach().cpu().numpy()
            target_i = target_outputs[i][target_care].detach().cpu().numpy()
            np.save("numpy_emb/source_layer{}".format(i), source_i)
            np.save("numpy_emb/target_layer{}".format(i), target_i)
        
        return source_outputs, target_outputs
    

    def get_candidate(self, source_outputs, target_outputs):
        List_S = self.get_similarity_matrices(source_outputs, target_outputs)[1:]
        source_candidates = []
        target_candidates = []
        if len(List_S) < 2:
            print("Current model doesn't support refinement for number of GCN layer smaller than 2")
            return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

        num_source_nodes = len(self.source_dataset.G.nodes())
        num_target_nodes = len(self.target_dataset.G.nodes())
        for i in range(min(num_source_nodes, num_target_nodes)):
            node_i_is_stable = True
            for j in range(len(List_S)):
                if List_S[j][i].argmax() != List_S[j-1][i].argmax() or List_S[j][i].max() < self.args.threshold_refine:
                    node_i_is_stable = False 
                    break
            if node_i_is_stable:
                tg_candi = List_S[-1][i].argmax()
                
                source_candidates.append(i)
                target_candidates.append(tg_candi)

        return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

