import os
import random
import torch
import warnings
import time
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
from utils.apt2021_pipeline import train_apt2021
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.autoencoder import build_model
from model.train import batch_level_train

warnings.filterwarnings('ignore')

def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader

def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset.lower()
    main_args.dataset = dataset_name
    default_max_epoch = 50
    if dataset_name == 'wget':
        main_args.num_hidden = 32
        default_max_epoch = 10
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    if main_args.max_epoch is None:
        main_args.max_epoch = default_max_epoch
    else:
        main_args.max_epoch = int(main_args.max_epoch)
    set_random_seed(42)

    if dataset_name == 'wget':
        batch_size = 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model = model.to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim)
        os.makedirs("./result", exist_ok=True)
        torch.save(model.state_dict(), "./result/checkpoint-{}.pt".format(dataset_name))
    else:
        if dataset_name == 'apt2021':
            default_max_epoch = 10
            train_apt2021(device=device, override_max_epoch=main_args.max_epoch)
            return
        
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model = model.to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        epoch_iter = tqdm(range(main_args.max_epoch))
        n_train = metadata['n_train']
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                step_start = time.time()
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                model.train()
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                step_elapsed = time.time() - step_start
                if main_args.log_interval > 0 and (((i + 1) % main_args.log_interval == 0) or (i + 1 == n_train)):
                    tqdm.write(
                        f"[train-step] epoch={epoch} graph={i + 1}/{n_train} "
                        f"loss={loss.item():.6f} nodes={g.num_nodes()} edges={g.num_edges()} "
                        f"time={step_elapsed:.2f}s"
                    )
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        os.makedirs("./result", exist_ok=True)
        torch.save(model.state_dict(), "./result/checkpoint-{}.pt".format(dataset_name))
        save_dict_path = './result/distance_save_{}.pkl'.format(dataset_name)
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
    return


if __name__ == '__main__':
    args = build_args()
    main(args)