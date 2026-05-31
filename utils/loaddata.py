import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os


class WgetDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(WgetDataset, self).__init__(name=name)
        if name == 'wget':
            path = './data/wget/final'
            num_graphs = 150
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 0 <= idx <= 24:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def load_rawdata(name):
    if name == 'wget':
        path = './data/wget'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:
            raw_data = WgetDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    else:
        raise NotImplementedError
    return raw_data


def load_batch_level_dataset(dataset_name):
    dataset = load_rawdata(dataset_name)
    graph, _ = dataset[0]
    node_feature_dim = 0
    for g, _ in dataset:
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item())
    edge_feature_dim = 0
    for g, _ in dataset:
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item())
    node_feature_dim += 1
    edge_feature_dim += 1
    full_dataset = [i for i in range(len(dataset))]
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim))

    return {'dataset': dataset,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            'e_feat': edge_feature_dim}


def transform_graph(g, node_feature_dim, edge_feature_dim):
    new_g = g.clone()
    if "attr" in new_g.ndata and "attr" in new_g.edata:
        return new_g
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def _resolve_entity_dataset_dir(path):
    key = path.lower()
    if key == "clearscope":
        return "./data/clearscope", key
    if key == "apt2021":
        return "./data/APT2021", key
    if key == "apt2024":
        return "./data/APT2024", key
    return "./data/" + path, key


def preload_clearscope_entity_level_dataset():
    from utils.clearscope_parser import build_clearscope_preprocessed

    build_clearscope_preprocessed(
        data_dir="./data/clearscope",
        benign_train_ratio=float(os.getenv("CLEARSCOPE_TRAIN_BENIGN_RATIO", "0.8")),
        max_benign=int(os.getenv("CLEARSCOPE_MAX_BENIGN", "0")),
        max_attack=int(os.getenv("CLEARSCOPE_MAX_ATTACK", "0")),
        max_edges_per_graph=int(os.getenv("CLEARSCOPE_MAX_EDGES_PER_GRAPH", "300000")),
        force=False,
    )


def preload_apt2024_entity_level_dataset():
    from utils.apt2024_parser import build_apt2024_preprocessed

    build_apt2024_preprocessed(
        data_dir="./data/APT2024",
        phase1_train_ratio=float(os.getenv("APT2024_TRAIN_BENIGN_RATIO", "0.8")),
        max_edges_per_graph=int(os.getenv("APT2024_MAX_EDGES_PER_GRAPH", "0")),
        force=False,
    )


def preload_apt2021_entity_level_dataset():
    from utils.apt2021_parser import build_apt2021_preprocessed

    build_apt2021_preprocessed(
        data_dir="./data/APT2021",
        window_seconds=int(os.getenv("APT2021_WINDOW_SECONDS", "600")),
        min_edges_per_graph=int(os.getenv("APT2021_MIN_EDGES_PER_GRAPH", "200")),
        max_train_windows=int(os.getenv("APT2021_MAX_TRAIN_WINDOWS", "0")),
        max_test_windows=int(os.getenv("APT2021_MAX_TEST_WINDOWS", "0")),
        max_edges_per_graph=int(os.getenv("APT2021_MAX_EDGES_PER_GRAPH", "0")),
        force=False,
    )


def preload_entity_level_dataset(path):
    data_dir, key = _resolve_entity_dataset_dir(path)
    if key == "clearscope":
        preload_clearscope_entity_level_dataset()
        return
    if key == "apt2021":
        preload_apt2021_entity_level_dataset()
        return
    if key == "apt2024":
        preload_apt2024_entity_level_dataset()
        return
    path = data_dir
    if os.path.exists(path + '/metadata.json'):
        pass
    else:
        print('transforming')
        train_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))]
        print('transforming')
        test_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))]
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))

        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_test_gs.append(g)
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_train_gs.append(g)
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + '/train{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + '/test{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)


def load_metadata(path):
    preload_entity_level_dataset(path)
    data_dir, _ = _resolve_entity_dataset_dir(path)
    with open(data_dir + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    preload_entity_level_dataset(path)
    data_dir, _ = _resolve_entity_dataset_dir(path)
    with open(data_dir + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data
