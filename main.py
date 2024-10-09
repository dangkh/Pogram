import argparse
import numpy as np
from torch import nn
from src.config import TrainConfig 
from src.ultis import *
from src.data_helper import prepare_preprocessed_data

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train( cfg ) -> None:
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    
    """
    logging.info("Prepare the dataset")
    prepare_preprocessed_data(cfg)

    # """
    # 1. Init Model
    # """
    logging.info("Initialize Model")
    # news_encoder = PLMBasedNewsEncoder(pretrained)
    # user_encoder = UserEncoder(hidden_size=hidden_size)
    # nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
    #     device, dtype=torch.bfloat16)

    model = load_model(cfg.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


    """
    2. Load Data & Create Dataset
    """
    # logging.info("Initialize Dataset")
    # train_data_dir = './data/MINDsmall_train'
    # train_news_df = read_news_df(os.path.join(train_data_dir, 'news.tsv'))
    # train_behavior_df = read_behavior_df(os.path.join(train_data_dir, 'behaviors.tsv'))
    # logging.info(train_behavior_df[0])
    # train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device)
    
    # test_data_dir = './data/MINDsmall_dev'
    # val_news_df = read_news_df(os.path.join(test_data_dir, 'news.tsv'))
    # val_behavior_df = read_behavior_df(os.path.join(test_data_dir, 'behaviors.tsv'))
    # eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logging.info("Training Start")
    # 
    """
    4. Evaluate model by Validation Dataset
    """
    logging.info("Evaluation")
    # metrics = evaluate(trainer.model, eval_dataset, device)
    # logging.info(metrics.dict())


def main(cfg: TrainConfig) -> None:
	set_random_seed(cfg.random_seed)
	train(cfg)
    # try:
    #     set_random_seed(cfg.random_seed)
    #     train(cfg)
    # except Exception as e:
    #     logging.error(e)

if __name__ == "__main__":
    main(TrainConfig)