import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle

import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph


class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.his_size)
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)
    
    
class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors

    def line_mapper(self, line, sum_num_news):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # ------------------ News Subgraph ---------------------
        top_k = len(click_id)
        click_idx = self.trans_to_nindex(click_id)  
        source_idx = click_idx     
        for _ in range(self.cfg.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.his_size-len(mapping_idx), 0), "constant", -1)

        
        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.entity_size:-3]  #[5, 5]
            candidate_neighbor_entity = np.zeros(((self.cfg.npratio+1) * self.cfg.entity_size, self.cfg.entity_neighbors), dtype=np.int64) # [5*5, 20]
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio+1, self.cfg.entity_size *self.cfg.entity_neighbors) # [5, 5*20]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
               sum_num_news+sub_news_graph.num_nodes

    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset: 
            subset = [0]
            
        subset = torch.tensor(subset, dtype=torch.long, device=device)
        
        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr, relabel_nodes=True, num_nodes=self.news_graph.num_nodes)
                    
        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k]+sum_num_nodes
    
    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))


                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        clicked_graphs, mappings ,candidates, labels, candidate_entity_list, entity_mask_list  = [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)


class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.his_size:]

        click_idx = self.trans_to_nindex(click_id)
        clicked_entity = self.news_entity[click_idx]  
        source_idx = click_idx     
        for _ in range(self.cfg.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

         # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index)*self.cfg.entity_size, self.cfg.entity_neighbors), dtype=np.int64)
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]
            
            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.entity_size *self.cfg.entity_neighbors)
       
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels
    
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]



def load_data(cfg, mode='train', model=None, local_rank=0):
    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir}

    # ------------- load news.tsv-------------
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    # ------------- load behaviors_np{X}.tsv --------------
    if mode == 'train':
        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv"
        if cfg.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            if cfg.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")


            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.use_entity:
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            dataset = TrainGraphDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors
            )
            dataloader = DataLoader(dataset, batch_size=None)
            
        else:
            dataset = TrainDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
            )

            dataloader = DataLoader(dataset,
                                    batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    pin_memory=True)
        return dataloader
    elif mode in ['val', 'test']:
        # convert the news to embeddings
        news_dataset = NewsDataset(news_input)
        news_dataloader = DataLoader(news_dataset,
                                     batch_size=int(cfg.batch_size * cfg.gpu_num))

        stacked_news = []
        with torch.no_grad():
            for news_batch in tqdm(news_dataloader, desc=f"[{local_rank}] Processing validation News Embedding"):
                if cfg.use_graph:
                    batch_emb = model.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                else:
                    batch_emb = model.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                stacked_news.append(batch_emb)
        news_emb = torch.cat(stacked_news, dim=0).cpu().numpy()   

        if cfg.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            if cfg.use_entity:
                # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            if mode == 'val':
                dataset = ValidGraphDataset(
                    filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    news_entity=news_input[:,-8:-3],
                    entity_neighbors=entity_neighbors
                )

            dataloader = DataLoader(dataset, batch_size=None)

        else:
            if mode == 'val':
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors_0.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )
            else:
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )

            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    # batch_size=int(cfg.batch_   size / cfg.gpu_num),
                                    # pin_memory=True, # collate_fn already puts data to GPU
                                    collate_fn=lambda b: collate_fn(b, local_rank))
        return dataloader


def collate_fn(tuple_list, local_rank):
    clicked_news = [x[0] for x in tuple_list]
    clicked_mask = [x[1] for x in tuple_list]
    candidate_news = [x[2] for x in tuple_list]
    clicked_index = [x[3] for x in tuple_list]
    candidate_index = [x[4] for x in tuple_list]

    if len(tuple_list[0]) == 6:
        labels = [x[5] for x in tuple_list]
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index, labels
    else:
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index