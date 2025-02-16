import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data
from src.data_load import *
from src.metrics import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--checknum', type=int, default=4, help=f'')
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



def check_modelPanel(model, cfg, mode = 'val'):

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
			graph_vec = model.gcn(graph_vec, edge_index)
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
			user_vecs = torch.stack([user_vecs, graph_vec], dim=1)
			user_vecs = model.loc_glob_att(user_vecs)

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
	
	return res
cfg = TrainConfig

logging.info("Start")
set_random_seed(cfg.random_seed)

logging.info("Initialize Model")
model, optimizer = load_model(cfg)
model = model.to(device)
print(model)
checkpoint = torch.load(f'./checkpoint/{args.checknum}use_graph{cfg.use_graph}_use_entity{cfg.use_graph}.pth', weights_only = False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore additional information
epoch = checkpoint['epoch']
loss = checkpoint['loss']
	
logging.info("Evaluation")
testRes = check_modelPanel(model, cfg, mode='val')
print(testRes)
logging.info("End")