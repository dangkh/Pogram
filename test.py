import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data
from src.data_load import *
from src.metrics import *
import warnings
import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--reprocess", action="store_true", help="Enable entity usage (default: False)")
parser.add_argument("--val_all", action="store_true", help="Enable validate all epoch (default: False)")
parser.add_argument('--checknum', type=int, default=4, help=f'')
parser.add_argument("--use_graph", action="store_true", help="Enable graph usage (default: False)")
parser.add_argument("--use_entity", action="store_true", help="Enable entity usage (default: False)")
parser.add_argument("--use_EnrichE", action="store_true", help="Enable EnrichE usage (default: False)")
parser.add_argument("--prototype", action="store_false", default=True, help="Enable prototype (default: True)")
parser.add_argument("--genAbs", action="store_true", help="Enable abstract generation (default: False)")
parser.add_argument("--absType", type=int, choices=[0, 1], default=0, help="Abstraction type: 0 for direct, 1 for via entity (default: 0)")
args = parser.parse_args()

def acc(y_true, y_hat):
	y_hat = torch.argmax(y_hat, dim=-1)
	tot = y_true.shape[0]
	hit = torch.sum(y_true == y_hat)
	return hit.data.float() * 1.0 / tot

def print_metrics(cnt, x):
	print(cnt, x)

def get_mean(arr):
	return [np.array(i).mean() for i in arr]

def get_sum(arr):
	return [np.array(i).sum() for i in arr]



def evaluate_modelPanel(model, cfg, mode = 'val'):
	model.eval()
	torch.set_grad_enabled(False)
	valid_dataloader = load_dataloader(cfg, mode, model)
	tasks = []
	AUC = []
	MRR = []
	nDCG5 = []
	nDCG10 = []
	for cnt, (graph_batch, [log_vecs, log_mask, news_vecs, labels]) in tqdm(enumerate(valid_dataloader)):
		log_vecs = log_vecs.cuda()
		log_mask = log_mask.cuda()

		if cfg.use_graph:
			graph_vec, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
			graph_vec = model.gnn1(graph_vec, edge_index)
			# graph_vec = model.relu(graph_vec)
			# graph_vec = model.gnn2(graph_vec, edge_index)
			# graph_vec = model.relu(graph_vec)
			graph_vec = model.gln(graph_vec)
			graph_vec = model.relu(graph_vec)
			graph_vec = model.glob_mean(graph_vec, batch)

		if cfg.use_entity:
			news_vecs = news_vecs.cuda()
			e_his = log_vecs[:,:,-5:].int()
			log_vecs = log_vecs[:,:,:-5]
			e_candi = news_vecs[:,:,-5:].int()
			news_vecs = news_vecs[:,:,:-5]

		if cfg.use_entity:
			e_his = model.entity_embedding_layer(e_his)
			e_candi = model.entity_embedding_layer(e_candi)
			e_his = model.entity_encoder(e_his, None)
			e_candi = model.entity_encoder(e_candi, None)
		
			user_vecs = model.user_att(torch.stack([log_vecs, e_his], dim=2).view(-1, 2, model.news_dim))
			user_vecs = user_vecs.view(-1, model.user_log_length, model.news_dim)
			news_vecs = model.candi_att(torch.stack([news_vecs, e_candi], dim=2).view(-1, 2, model.news_dim))
			news_vecs = news_vecs.unsqueeze(0).detach().cpu().numpy()

		else:
			user_vecs = log_vecs.view(-1, model.user_log_length, model.news_dim)

		
		user_vecs = model.user_encoder(user_vecs, log_mask)
		if cfg.use_graph:
			graph_vec = model.loc_glob_att(graph_vec, user_vecs, user_vecs)
			graph_vec = model.graph2newsDim(graph_vec).view(-1, model.news_dim)
			graph_vec = model.relu(graph_vec)
			user_vecs = torch.stack([user_vecs, graph_vec], dim=1)
			user_vecs = model.loc_glob_att2(user_vecs)

		user_vecs = user_vecs.detach().cpu().numpy()
		
		labels = labels.detach().cpu().numpy()
		for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):
			tmp = np.mean(label)
			if tmp == 0 or tmp == 1:
				continue

			score = np.dot(news_vec, user_vec)
			auc = roc_auc_score(label, score)
			mrr = mrr_score(label, score)
			ndcg5 = ndcg_score(label, score, k=5)
			ndcg10 = ndcg_score(label, score, k=10)

			AUC.append(auc)
			MRR.append(mrr)
			nDCG5.append(ndcg5)
			nDCG10.append(ndcg10)
	reduced_auc, reduced_mrr, reduced_ndcg5, reduced_ndcg10 = get_mean([AUC, MRR, nDCG5, nDCG10])
	res = {
		"auc": reduced_auc,
		"mrr": reduced_mrr,
		"ndcg5": reduced_ndcg5,
		"ndcg10": reduced_ndcg10,
	}
	wandb.log(res)
	print(res)
	
	return res

def updateModel(model, val_epoch):
	checkpoint = torch.load(f'./checkpoint/{val_epoch}_g{cfg.use_graph}_e1{cfg.use_entity}_e2{cfg.use_EnrichE}_abs{cfg.genAbs}_absType{cfg.absType}.pth', 
		weights_only = False)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	# Restore additional information
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']	
	return model





cfg = TrainConfig()
cfg.update(args)
if cfg.genAbs:
	cfg.title_size = 50
wandb.init(
	# set the wandb project where this run will be logged
	project="pogram",
	# track hyperparameters and run metadata
	config={
	"learning_rate": cfg.learning_rate,
	"epochs": cfg.epochs,
	"use_graph": cfg.use_graph,
	"use_entity" :cfg.use_entity,
	"use_EnrichE" :cfg.use_EnrichE,
	"prototype" :cfg.prototype,
	"genAbs" :cfg.genAbs,
	"absType": cfg.absType,
	"his_size": cfg.his_size,
	"history_size": cfg.history_size,
	"batch_size": cfg.batch_size,
	"num_neighbors":cfg.num_neighbors,
	"head_num": cfg.head_num,
	"k_hops" :cfg.k_hops,
	"head_dim": cfg.head_dim,
	"entity_emb_dim" :cfg.entity_emb_dim,
	"entity_neighbors": cfg.entity_neighbors,
	"checknum": args.checknum
	})
logging.info("Start")
set_random_seed(cfg.random_seed)

logging.info("Initialize Model")
model, optimizer = load_model(cfg)
model = model.to(device)
logging.info("Evaluation")
if args.val_all:
	for e in range(cfg.epochs):
		model = updateModel(model, e)
		evaluate_modelPanel(model, cfg, mode='test')
else:
	val_epoch = args.checknum
	model = updateModel(model, val_epoch)
	evaluate_modelPanel(model, cfg, mode='test')


logging.info("End")
wandb.finish()