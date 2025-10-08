import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.prov2hyper import prov_graphml_to_hypergraph

class Apt2021Dataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        H, X, names, timestamp, flow_order, edge_index = prov_graphml_to_hypergraph(self.paths[idx])
        is_flow = torch.tensor([nm.startswith("flow_") for nm in names],
                               dtype=torch.bool)
        return Data(x=X, incidence=H, is_flow=is_flow,
                    timestamp=timestamp, flow_order=flow_order,
                    edge_index=edge_index)
