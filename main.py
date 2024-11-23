import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data
from src.data_load import *
from src.metrics import *

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train(model, optimizer, train_dataloader, device, cfg)
def train(model, optimizer, dataloader, cfg, early_stopping):
    model.train()
    torch.set_grad_enabled(True)
    for e in range(cfg.epochs):
        print(f"Training at Epoch: {e}")
        sum_loss = torch.zeros(1).to(device)
        sum_auc = torch.zeros(1).to(device)

        for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels) \
                in enumerate(tqdm(dataloader,
                                  total=int(cfg.epochs * (236344 // cfg.batch_size + 1)),
                                  desc="Training"), start=1):
            subgraph = subgraph.to(device, non_blocking=True)
            mapping_idx = mapping_idx.to(device, non_blocking=True)
            candidate_news = candidate_news.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            candidate_entity = candidate_entity.to(device, non_blocking=True)
            entity_mask = entity_mask.to(device, non_blocking=True)

            
            bz_loss, y_hat = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels)
                
            bz_loss.backward()
            optimizer.step()
            
            sum_loss += bz_loss.data.float()
            sum_auc += area_under_curve(labels, y_hat)
            # ---------------------------------------- Training Log
            
        res = val(model, cfg)
        print(res)
        model.train()
        

        early_stop, get_better = early_stopping(res['auc'])
        if early_stop:
            print("Early Stop.")
            break
        elif get_better:
            print(f"Better Result!")
            save_model(cfg, model, optimizer, f"_auc{res['auc']}")



def val(model, cfg):
    model.eval()
    dataloader = load_data(cfg, mode='val', model=model, local_rank=device)
    tasks = []
    with torch.no_grad():
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels) \
                in enumerate(tqdm(dataloader, "Validating")):
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(device, non_blocking=True)
            candidate_entity = candidate_entity.to(device, non_blocking=True)
            entity_mask = entity_mask.to(device, non_blocking=True)
            clicked_entity = clicked_entity.to(device, non_blocking=True)

            scores = model.validation_process(subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask)
            tasks.append((labels.tolist(), scores))

    results = map(cal_metric, tasks)
    results = list(results)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier

    reduced_auc = np.mean(val_auc)
    reduced_mrr = np.mean(val_mrr)
    reduced_ndcg5 = np.mean(val_ndcg5)
    reduced_ndcg10 = np.mean(val_ndcg10)

    res = {
        "auc": reduced_auc,
        "mrr": reduced_mrr,
        "ndcg5": reduced_ndcg5,
        "ndcg10": reduced_ndcg10,
    }
    
    return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def executor(cfg) -> None:
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    
    """
    logging.info("Prepare the dataset")
    prepare_preprocessed_data(cfg)
    """
    1. Load Data & Create Dataset
    """
    train_dataloader = load_data(cfg, mode='train', local_rank=device)

    # stop
    """
    2. Init Model
    """
    logging.info("Initialize Model")
    model = load_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # print(model)
    logging.info(f"NUMBER parameters: {count_parameters(model)}")
    early_stopping = EarlyStopping(cfg.early_stop_patience)

    # stop
    """
    3. Train
    """
    logging.info("Training Start")
    # for _ in tqdm(range(1, cfg.num_epochs + 1), desc="Epoch"):
    train(model, optimizer, train_dataloader, cfg, early_stopping)
    # 
    """
    4. Evaluate model by Validation Dataset
    """
    logging.info("Evaluation")
    # metrics = evaluate(trainer.model, eval_dataset, device)
    # logging.info(metrics.dict())


def main(cfg: TrainConfig) -> None:
	set_random_seed(cfg.random_seed)
	executor(cfg)
    # try:
    #     set_random_seed(cfg.random_seed)
    #     train(cfg)
    # except Exception as e:
    #     logging.error(e)

if __name__ == "__main__":
    main(TrainConfig)