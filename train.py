import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data
from src.data_load import *
from src.metrics import *
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_modelPanel(model, optimizer, dataloader, cfg):
	model = model.to(device)
	model.train()
	torch.set_grad_enabled(True)
	for ep in range(cfg.epochs):
		loss = 0.0
		accuary = 0.0
		print("EPOCH: " + str(ep))
		for cnt, (g, [log_ids, log_mask, input_ids, targets]) in tqdm(enumerate(dataloader)):
			log_ids = log_ids.to(device)
			log_mask = log_mask.to(device)
			input_ids = input_ids.to(device)
			targets = targets.to(device)

			bz_loss, y_hat = model([g, log_ids, log_mask, input_ids, targets])
			loss += bz_loss.data.float()
			accuary += acc(targets, y_hat)
			optimizer.zero_grad()
			bz_loss.backward()
			optimizer.step()

		# eval_acc = evaluate_modelPanel(model, cfg)
		print(loss, accuary)
		model.train()

		
		checkpoint = {
			'epoch': cfg.epochs,  
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,  # Save loss or any other metric
		}
		# Save the checkpoint
		torch.save(checkpoint, f'./checkpoint/use_graph{cfg.use_graph}_use_entity{cfg.use_graph}.pth')

cfg = TrainConfig

logging.info("Start")
set_random_seed(cfg.random_seed)
"""
0. Definite Parameters & Functions

"""
logging.info("Prepare the dataset")
# if using Enriched Entity, make sure that reprocess setting is True
prepare_preprocessed_data(cfg)
train_dataloader = load_dataloader(cfg, mode='train')

logging.info("Initialize Model")
model, optimizer = load_model(cfg)
print(model)
logging.info("Training Start")
train_modelPanel(model, optimizer, train_dataloader, cfg)
logging.info("End")