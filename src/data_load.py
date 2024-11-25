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
from torch_geometric.loader import DataLoader as GraphDataLoader

import random

	
class Panel_TrainDataset(Dataset):
	def __init__(self, filename, news_index, news_combined, cfg, neighbor_dict, news_graph):
		super(Panel_TrainDataset).__init__()
		self.filename = filename
		self.news_index = news_index
		self.news_combined = news_combined
		self.user_log_length = cfg.his_size
		self.npratio = cfg.npratio
		self.cfg = cfg
		self.neighbor_dict = neighbor_dict
		self.news_graph = news_graph
		self.news_graph.x = self.news_graph.x.float()
		self.prepare()

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

	def build_k_hop(self, click_doc):
		click_idx = [x for x in click_doc]
		source_idx = [x for x in click_doc]
		for _ in range(self.cfg.k_hops) :
			current_hop_idx = []
			for news_idx in source_idx:
				current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.num_neighbors])
			source_idx = current_hop_idx
			click_idx.extend(current_hop_idx)
		return list(set(click_idx))
		
	def prepare(self):
		self.preprocessDT = []
		with open(self.filename) as f:
			for line in tqdm(f):
				g, dt = self.line_mapper(line)
				if len(g) == 0:
					continue
				self.preprocessDT.append([g,dt])
				# if len(self.preprocessDT) > 10000:
				# 	break
	
	def line_mapper(self, line):
		line = line.strip().split('\t')
		click_docs = line[3].split()
		sess_pos = line[4].split()
		sess_neg = line[5].split()
		click_docs = self.trans_to_nindex(click_docs)

		# build sub-graph news
		# k_hops_click = self.build_k_hop(click_docs)
		k_hops_click = click_docs

		# build sub-graph entity
		click_docs, log_mask = self.pad_to_fix_len(click_docs, self.user_log_length)
		user_feature = self.news_combined[click_docs]

		pos = self.trans_to_nindex(sess_pos)
		neg = self.trans_to_nindex(sess_neg)

		label = random.randint(0, self.npratio)
		sample_news = neg[:label] + pos + neg[label:]
		news_feature = self.news_combined[sample_news]
		return k_hops_click, [torch.from_numpy(user_feature), torch.from_numpy(log_mask), \
		torch.from_numpy(news_feature), torch.tensor(label)]

	def __getitem__(self, idx):
		k_hops_click, dt =  self.preprocessDT[idx]
		# subemb = self.news_graph.x[k_hops_click]
		# sub_edge_index, sub_edge_attr = subgraph(k_hops_click, self.news_graph.edge_index, self.news_graph.edge_attr, \
		#                                          relabel_nodes=True, num_nodes=self.news_graph.num_nodes)
		# sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr).cuda()
		return "", dt

	def __len__(self):
		return len(self.preprocessDT)

class Panel_ValidDataset(Panel_TrainDataset):
	def __init__(self, filename, news_index, news_score, cfg, neighbor_dict, news_graph):
		super(Panel_ValidDataset).__init__()
		self.filename = filename
		self.news_index = news_index
		self.news_score = news_score
		self.user_log_length = cfg.his_size
		self.npratio = cfg.npratio
		self.cfg = cfg
		self.neighbor_dict = neighbor_dict
		self.news_graph = news_graph
		self.news_graph.x = self.news_graph.x.float()
		self.prepare()


	def line_mapper(self, line):
		line = line.strip().split('\t')
		click_docs = line[3].split()
		
		candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
		label = [int(i.split('-')[1]) for i in line[4].split()]
		
		click_docs = self.trans_to_nindex(click_docs)

		# build sub-graph
		k_hops_click = self.build_k_hop(click_docs)
		
		click_docs, log_mask = self.pad_to_fix_len(click_docs, self.user_log_length)
		user_feature = self.news_score[click_docs]

		news_feature = self.news_score[candidate_news]
		
		return k_hops_click,  [torch.from_numpy(user_feature), torch.from_numpy(log_mask), \
		torch.from_numpy(news_feature), torch.tensor(label)]



class NewsDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return self.data.shape[0]


# news_dict = {news_id: str, news_id: int}
# nltk_token_news = {news_id: int, news_feature: list} news_feature = [tokenized title, category, subcate, newsid, entities]
# news_graph = [edge: pair, edge_index: list keys_pair, edge_attr: list edge_weight]
# news_neighbors_dict
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

			dataloader = DataLoader(dataset, batch_size=32)
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


def load_dataloaderEntity(cfg, mode='train', model=None):
	data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir + '_val'}

	# ------------- load news.tsv-------------
	news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

	news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
	# ------------- load behaviors_np{X}.tsv --------------
	news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))
	news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")
	if mode == 'train':
		target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv"
		# if cfg.use_graph:

		#     if cfg.directed is False:
		#         news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
		#     print(f"[{mode}] News Graph Info: {news_graph}")


		# if cfg.use_entity:
		#     entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
		#     total_length = sum(len(lst) for lst in entity_neighbors.values())
		#     print(f"[{mode}] entity_neighbor list Length: {total_length}")
		# else:
		#     entity_neighbors = None

		dataset = Panel_TrainDataset(
			filename=target_file,
			news_index=news_index,
			news_combined=news_input,
			cfg=cfg,
			neighbor_dict=news_neighbors_dict,
			news_graph=news_graph
		)
		dataloader = GraphDataLoader(dataset, batch_size=128)
	elif mode in ['val', 'test']:
		# convert the news to embeddings
		news_dataset = NewsDataset(news_input)
		news_dataloader = DataLoader(news_dataset,  batch_size= 128)

		news_scoring = []
		with torch.no_grad():
			for input_ids in tqdm(news_dataloader):
				e_lis = input_ids[:,-5:]
				input_ids = input_ids[:,:-5].cuda()
				news_vec = model.news_encoder(input_ids)
				news_vec = news_vec.to(torch.device("cpu"))
				news_vec = torch.concatenate((news_vec, e_lis),-1)
				news_vec = news_vec.detach().numpy()
				news_scoring.extend(news_vec)

		news_scoring = np.array(news_scoring)

		# if cfg.use_graph:
		#     news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

		#     news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

		#     if cfg.directed is False:
		#         news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
		#     print(f"[{mode}] News Graph Info: {news_graph}")

		#     if cfg.use_entity:
		#         # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
		#         entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
		#         total_length = sum(len(lst) for lst in entity_neighbors.values())
		#         print(f"[{mode}] entity_neighbor list Length: {total_length}")
		#     else:
		#         entity_neighbors = None
		if mode == 'val':
			dataset = Panel_ValidDataset(
				filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv",
				news_index=news_index,
				news_score=news_scoring,
				cfg=cfg,
				neighbor_dict=news_neighbors_dict,
				news_graph=news_graph
			)

			dataloader = GraphDataLoader(dataset, batch_size=1)

		else:
			dataset = Panel_ValidDataset(
					filename=Path(data_dir[mode]) / f"behaviors.tsv",
					news_index=news_index,
					news_score=news_scoring,
					cfg=cfg,
					neighbor_dict=news_neighbors_dict,
					news_graph=news_graph
				)

			dataloader = GraphDataLoader(dataset, batch_size=1)
		

	return dataloader

