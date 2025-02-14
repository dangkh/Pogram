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
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
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
		if self.cfg.use_graph:
			self.news_graph = news_graph
			self.news_graph.x = self.news_graph.x.float()
		self.listPrep = []
		self.prepDatabyUser = []
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
		num_line = len(open(self.filename, encoding='utf-8').readlines())
		with open(self.filename) as f:
			for line in tqdm(f, total=num_line):
				uid, dt = self.line_mapper(line)
				if len(uid) == 0:
					continue
				self.preprocessDT.append([uid,dt])
				if self.cfg.prototype and (len(self.preprocessDT) > 10000):
					break
	
	def line_mapper(self, line):
		line = line.strip().split('\t')
		uid = line[1]
		if uid not in self.listPrep:
			click_docs = line[3].split()
			click_docs = self.trans_to_nindex(click_docs)
			k_hops_click = click_docs
			if self.cfg.use_graph:
				k_hops_click = self.build_k_hop(click_docs)
			click_docs, log_mask = self.pad_to_fix_len(click_docs, self.user_log_length)
			user_feature = self.news_combined[click_docs]

			if self.cfg.use_graph:
				subemb = self.news_graph.x[k_hops_click]
				sub_edge_index, sub_edge_attr = subgraph(k_hops_click, self.news_graph.edge_index, self.news_graph.edge_attr, \
														relabel_nodes=True, num_nodes=self.news_graph.num_nodes)
				sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
				self.prepDatabyUser.append([k_hops_click, sub_news_graph, user_feature, log_mask])
			else:
				self.prepDatabyUser.append([k_hops_click, user_feature, log_mask])
			self.listPrep.append(uid)
		else:
			if self.cfg.use_graph:
				k_hops_click, _, _, _ = self.prepDatabyUser[self.listPrep.index(uid)]
			else:	
				k_hops_click, _, _ = self.prepDatabyUser[self.listPrep.index(uid)]
		
		sess_pos = line[4].split()
		sess_neg = line[5].split()
		pos = self.trans_to_nindex(sess_pos)
		neg = self.trans_to_nindex(sess_neg)

		label = random.randint(0, self.npratio)
		sample_news = neg[:label] + pos + neg[label:]
		news_feature = self.news_combined[sample_news]

		return k_hops_click, [uid, torch.from_numpy(news_feature), torch.tensor(label)]


	def __getitem__(self, idx):
		k_hops_click, [uid, news_feature, label] =  self.preprocessDT[idx]
		sub_news_graph = []
		if self.cfg.use_graph:
			_, sub_news_graph, user_feature, log_mask = self.prepDatabyUser[self.listPrep.index(uid)]
			sub_news_graph = sub_news_graph.to(device)
		else:	
			_, user_feature, log_mask = self.prepDatabyUser[self.listPrep.index(uid)]
		return sub_news_graph, [torch.from_numpy(user_feature), torch.from_numpy(log_mask), news_feature, label]

	def __len__(self):
		return len(self.preprocessDT)

class Panel_ValidDataset(Panel_TrainDataset):
	def __init__(self, filename, news_index, news_score, cfg, neighbor_dict, news_graph, mode = "val"):
		super(Panel_ValidDataset).__init__()
		self.filename = filename
		self.news_index = news_index
		self.news_score = news_score
		self.user_log_length = cfg.his_size
		self.npratio = cfg.npratio
		self.cfg = cfg
		self.neighbor_dict = neighbor_dict
		if cfg.use_graph:
			self.news_graph = news_graph
			self.news_graph.x = self.news_graph.x.float()
		self.file = open(self.filename, encoding='utf-8').readlines()
		self.numline = len(self.file)
		self.listPrep = []

		self.prepDatabyUser = []
		self.mode = mode

	def line_mapper(self, line):
		line = line.strip().split('\t')
		uid = line[1]
		
		click_docs = line[3].split()
		click_docs = self.trans_to_nindex(click_docs)
		k_hops_click = click_docs
		click_docs, log_mask = self.pad_to_fix_len(click_docs, self.user_log_length)
		user_feature = self.news_score[click_docs]

		candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
		if self.mode == "test":
			label = np.array([0 for i in line[4].split()])
		else:
			label = np.array([int(i.split('-')[1]) for i in line[4].split()])
		news_feature = self.news_score[candidate_news]

		return k_hops_click, [uid, torch.from_numpy(news_feature), torch.tensor(label), user_feature, log_mask]

	def __getitem__(self, idx):

		k_hops_click, [uid, news_feature, label, user_feature, log_mask] = self.line_mapper(self.file[idx])

		sub_news_graph = []
		# if self.cfg.use_graph:
		# 	_, sub_news_graph, user_feature, log_mask = self.prepDatabyUser[self.listPrep.index(uid)]
		# 	sub_news_graph = sub_news_graph.to(device)
		# else:	
		# 	_, user_feature, log_mask = self.prepDatabyUser[self.listPrep.index(uid)]

		return sub_news_graph, [torch.from_numpy(user_feature), torch.from_numpy(log_mask), news_feature, label]

	def __len__(self):
		return self.numline



class NewsDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return self.data.shape[0]

def load_dataloader(cfg, mode='train', model=None):
	data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir + '_test'}

	# ------------- load news.tsv-------------
	news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

	news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
	# ------------- load behaviors_np{X}.tsv --------------
	news_neighbors_dict  = []
	news_graph = []
	if cfg.use_graph:
		news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))
		news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt", weights_only=False)
	if mode == 'train':
		target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_0.tsv"
		if cfg.use_graph:
			if cfg.directed is False:
				news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
				print(f"[{mode}] News Graph Info: {news_graph}")

		# if cfg.use_entity_global:
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
		dataloader = GraphDataLoader(dataset, batch_size=cfg.batch_size)
	elif mode in ['val', 'test']:
		# convert the news to embeddings
		news_dataset = NewsDataset(news_input)
		news_dataloader = DataLoader(news_dataset,  batch_size= 128)

		news_scoring = []
		with torch.no_grad():
			for input_ids in tqdm(news_dataloader):
				if cfg.use_entity:
					e_lis = input_ids[:,-5:]
					input_ids = input_ids[:,:-5].cuda()
				else:
					input_ids = input_ids.cuda()
				news_vec = model.news_encoder(input_ids)
				news_vec = news_vec.to(torch.device("cpu"))
				if cfg.use_entity:
					news_vec = torch.concatenate((news_vec, e_lis),-1)
				news_vec = news_vec.detach().numpy()
				news_scoring.extend(news_vec)

		news_scoring = np.array(news_scoring)

		if cfg.use_graph:
			if cfg.directed is False:
				news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
			print(f"[{mode}] News Graph Info: {news_graph}")

		#     if cfg.use_entity_global:
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
					news_graph=news_graph,
					mode = "test"
				)

			dataloader = GraphDataLoader(dataset, batch_size=1)
		

	return dataloader
