import torch
import torch.nn as nn
from ..base.layers import *
from torch_geometric.nn import Sequential

class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.entity_dim = cfg.entity_emb_dim
        self.news_dim = 400

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, int(self.entity_dim / cfg.head_dim), cfg.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(self.entity_dim, cfg.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Linear(self.entity_dim, self.news_dim),
            nn.LeakyReLU(0.2),
        ])


    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity, _ = entity_input.shape

        if entity_mask is not None:
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask.view(batch_size*num_news, num_entity)).view(batch_size, num_news, self.news_dim)
        else:
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), None).view(batch_size, num_news, self.news_dim)

        return result

class GlobalEntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.entity_dim = cfg.entity_emb_dim
        self.news_dim = cfg.head_num * cfg.head_dim

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, cfg.head_num, cfg.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(cfg.head_num * cfg.head_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(cfg.head_num * cfg.head_dim, cfg.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(cfg.head_num * cfg.head_dim),
        ])


    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity,_ = entity_input.shape
        if entity_mask is not None:
            entity_mask = entity_mask.view(batch_size*num_news, num_entity)

        result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask).view(batch_size, num_news, self.news_dim)

        return result