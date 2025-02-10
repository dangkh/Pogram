import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data
from src.data_load import *
from src.metrics import *
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb
parser = argparse.ArgumentParser()
parser.add_argument("--reprocess", action="store_true", help="Enable entity usage (default: False)")
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


def train_modelPanel(model, optimizer, dataloader, cfg):
	model = model.to(device)
	model.train()
	torch.set_grad_enabled(True)
	for ep in range(cfg.epochs):
		loss = 0.0
		accuary = 0.0
		print("EPOCH: " + str(ep))
		for cnt, (g, [mapId, log_ids, log_mask, input_ids, targets]) in tqdm(enumerate(dataloader)):
			log_ids = log_ids.to(device)
			log_mask = log_mask.to(device)
			input_ids = input_ids.to(device)
			targets = targets.to(device)

			bz_loss, y_hat = model([g, mapId, log_ids, log_mask, input_ids, targets])
			loss += bz_loss.data.float()
			accuary += acc(targets, y_hat)
			optimizer.zero_grad()
			bz_loss.backward()
			optimizer.step()

		# eval_acc = evaluate_modelPanel(model, cfg)
		print(loss, accuary)
		wandb.log({"acc": accuary, "loss": loss})
		model.train()

		
		checkpoint = {
			'epoch': cfg.epochs,  
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,  # Save loss or any other metric
		}
		# Save the checkpoint
		torch.save(checkpoint, f'./checkpoint/{ep}_g{cfg.use_graph}_e1{cfg.use_entity}_e2{cfg.use_EnrichE}_abs{cfg.genAbs}_absType{cfg.absType}.pth')


cfg = TrainConfig()

cfg.update(args)
if cfg.genAbs:
	cfg.title_size = 50
os.environ["WANDB_MODE"]="offline"
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
    })
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
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,}")  # Format with commas for readability
print(f"Trainable Parameters: {trainable_params:,}")

train_modelPanel(model, optimizer, train_dataloader, cfg)

logging.info("End")
wandb.finish()