import os
import random
import numpy as np
import torch
from typing import Callable
import logging
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
import src.models.panel as panel
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_random_seed(random_seed: int = 42) -> None:
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True


class RecMetrics(BaseModel):
    ndcg_at_10: float
    ndcg_at_5: float
    auc: float
    mrr: float


class RecEvaluator:
    """Implementation of evaluation metrics calculation. The evaluation metrics used are based on Wu et al.'s approach
    ref: https://aclanthology.org/2020.acl-main.331.pdf
    """

    @classmethod
    def evaluate_all(cls, y_true: np.ndarray, y_score: np.ndarray) -> RecMetrics:
        return RecMetrics(
            **{
                "ndcg_at_10": cls.ndcg_score(y_true, y_score, 10),
                "ndcg_at_5": cls.ndcg_score(y_true, y_score, 5),
                "auc": roc_auc_score(y_true, y_score),
                "mrr": cls.mrr_score(y_true, y_score),
            }
        )

    @classmethod
    def dcg_score(cls, y_true: np.ndarray, y_score: np.ndarray, K: int = 5) -> float:
        discounts = np.log2(np.arange(len(y_true)) + 2)[:K]

        y_score_rank = np.argsort(y_score)[::-1]
        top_kth_y_true = np.take(y_true, y_score_rank)[:K]
        gains = 2**top_kth_y_true - 1

        return np.sum(gains / discounts)

    @classmethod
    def ndcg_score(cls, y_true: np.ndarray, y_score: np.ndarray, K: int = 5) -> float:
        best = cls.dcg_score(y_true, y_true, K)
        actual = cls.dcg_score(y_true, y_score, K)
        return actual / best

    @classmethod
    def mrr_score(cls, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_score_rank = np.argsort(y_score)[::-1]
        y_true_sorted_by_y_score = np.take(y_true, y_score_rank)
        rr_score = y_true_sorted_by_y_score / np.arange(1, len(y_true) + 1)
        return np.sum(rr_score) / np.sum(y_true)

def load_pretrain_emb(embedding_file_path, target_dict, target_dim):
    embedding_matrix = np.zeros(shape=(len(target_dict) + 1, target_dim))
    have_item = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                itme = line[0].decode()
                if itme in target_dict:
                    index = target_dict[itme]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_item.append(itme)
    print('-----------------------------------------------------')
    print(f'Dict length: {len(target_dict)}')
    print(f'Have words: {len(have_item)}')
    miss_rate = (len(target_dict) - len(have_item)) / len(target_dict) if len(target_dict) != 0 else 0
    print(f'Missing rate: {miss_rate}')
    return embedding_matrix

def load_model(cfg):
    framework = getattr(panel, cfg.model_name)

    if cfg.use_entity:
        entity_dict = pickle.load(open(Path(cfg.data_dir + '_val') / "entity_dict.bin", "rb"))
        entity_emb_path = Path(cfg.data_dir + '_val') / "combined_entity_embedding.vec"
        entity_emb = load_pretrain_emb(entity_emb_path, entity_dict, 100)
    else:
        entity_emb = None

    word_dict = pickle.load(open(Path(cfg.data_dir + '_train') / "word_dict.bin", "rb"))
    glove_emb = load_pretrain_emb(cfg.glove_path, word_dict, cfg.word_emb_dim)
    model = framework(cfg, glove_emb=glove_emb, entity_emb=entity_emb)

    return model