import json
import os
import argparse
from scipy.io import loadmat
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from input.data_preprocess import DataPreprocess

import utils.graph_utils as graph_utils


class Dataset:
    """
    this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
    G.json, id2idx.json, features.npy (optional)

    Arguments:
    - data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._load_G()
        self._load_id2idx()
        self._load_features()
        # self.load_edge_features()
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))

    def _load_G(self):
        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        self.G = json_graph.node_link_graph(G_data)
        if type(self.G.nodes()[0]) is int:
            mapping = {k: str(k) for k in self.G.nodes()}
            self.G = nx.relabel_nodes(self.G, mapping)

    def _load_id2idx(self):
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        conversion = type(self.G.nodes()[0])
        self.id2idx = {}
        id2idx = json.load(open(id2idx_file))
        for k, v in id2idx.items():
            self.id2idx[conversion(k)] = v

    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None
        return self.features

    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self, sparse=False):
        return graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False)

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)

    def check_id2idx(self):
        # print("Checking format of dataset")
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        # print("Pass")
        return True




class SynDataset:
    def __init__(self, num_nodes, p_create_edge, num_feats=0, seed=1, from_graph=None, num_del=0):
        if from_graph is None:
            self.G = nx.generators.random_graphs.gnp_random_graph(num_nodes, p_create_edge, seed=seed)
            self.id2idx = {id: i for i, id in enumerate(self.G.nodes())}
            if num_feats > 0:
                self.features = np.zeros((len(self.G.nodes()), num_feats))
                for i in range(self.features.shape[0]):
                    self.features[i][np.random.randint(0, num_feats)] = 1
            else:
                self.features = None
        else:
            self.G = from_graph.G.copy()
            if num_del > 0:
                count_del = 0
                self.considernodes = []
                self.considernodes2 = []
                for node in self.G.nodes():
                    if len(self.G.neighbors(node)) > 4:
                        for node2 in self.G.neighbors(node):
                            if len(self.G.neighbors(node2)) > 2:
                                self.G.remove_edge(node, node2)
                                self.considernode = node
                                self.considernode2 = node2
                                self.considernodes.append(node)
                                self.considernodes2.append(node2)
                                count_del += 1
                                if count_del == num_del:
                                    break
                            if count_del == num_del:
                                break
                    if count_del == num_del:
                        break
            array = np.arange(len(self.G.nodes()))
            np.random.shuffle(array)
            
            self.id2idx = {id: array[i] for i, id in enumerate(self.G.nodes())}
            if num_feats > 0:
                self.features = from_graph.features[array]
            self.groundtruth = {array[i]:i for i in range(len(from_graph.G.nodes()))}
            self.groundtruth_matrix = np.zeros((len(self.G.nodes()), len(self.G.nodes())))
            self.groundtruth_matrix[array, np.arange(len(self.G.nodes()))] = 1

        self.edge_features = None
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))
        

    def get_adjacency_matrix(self):
        return graph_utils.construct_adjacency(self.G, self.id2idx)

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)




def parse_args():
    parser = argparse.ArgumentParser(description="Test loading dataset")
    parser.add_argument('--source_dataset', default="/home/trunght/dataspace/graph/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="/home/trunght/dataspace/graph/douban/offline/graphsage/")
    parser.add_argument('--groundtruth', default="/home/trunght/dataspace/graph/douban/dictionaries/groundtruth")
    parser.add_argument('--output_dir', default="/home/trunght/dataspace/graph/douban/statistics/")
    return parser.parse_args()

def main(args):    
    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, "dict")
    DataPreprocess.evaluateDataset(source_dataset, target_dataset, groundtruth, args.output_dir)





if __name__ == "__main__":
    args = parse_args()
    main(args)
