import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv

from .base.layers import *
from .component.user_encoder import UserEncoder
from .component.entity_encoder import EntityEncoder
from .component.news_encoder import *


# class GLORY(nn.Module):
#     def __init__(self, cfg, glove_emb=None, entity_emb=None):
#         super().__init__()

#         self.cfg = cfg
#         self.use_entity = cfg.use_entity

#         self.news_dim =  cfg.head_num * cfg.head_dim
#         self.entity_dim = cfg.entity_emb_dim

#         # -------------------------- Model --------------------------
#         # News Encoder
#         self.local_news_encoder = NewsEncoder(cfg, glove_emb)

#         # GCN
#         self.global_news_encoder = Sequential('x, index', [
#             (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),'x, index -> x'),
#         ])
#         # Entity
#         if self.use_entity:
#             pretrain = torch.from_numpy(entity_emb).float()
#             self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

#             self.local_entity_encoder = Sequential('x, mask', [
#                 (self.entity_embedding_layer, 'x -> x'),
#                 (EntityEncoder(cfg), 'x, mask -> x'),
#             ])

#             self.global_entity_encoder = Sequential('x, mask', [
#                 (self.entity_embedding_layer, 'x -> x'),
#                 (GlobalEntityEncoder(cfg), 'x, mask -> x'),
#             ])
#         # Click Encoder
#         self.click_encoder = ClickEncoder(cfg)

#         # User Encoder
#         self.user_encoder = UserEncoder(cfg)
        
#         # Candidate Encoder
#         self.candidate_encoder = CandidateEncoder(cfg)

#         # click prediction
#         self.click_predictor = DotProduct()
#         self.loss_fn = NCELoss()


#     def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, label=None):
#         # -------------------------------------- clicked ----------------------------------
#         mask = mapping_idx != -1
#         mapping_idx[mapping_idx == -1] = 0

#         batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
#         clicked_entity = subgraph.x[mapping_idx, -8:-3]

#         # News Encoder + GCN
#         x_flatten = subgraph.x.view(1, -1, token_dim)
#         x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)

#         # graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)

#         clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)
#         # clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)

#         # Attention pooling
#         if self.use_entity:
#             clicked_entity = self.local_entity_encoder(clicked_entity, None)
#         else:
#             clicked_entity = None

#         # clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
#         clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_origin_emb, clicked_entity)
#         user_emb = self.user_encoder(clicked_total_emb, mask)

#         # ----------------------------------------- Candidate------------------------------------
#         cand_title_emb = self.local_news_encoder(candidate_news)                                      # [8, 5, 400]
#         if self.use_entity:
#             origin_entity, neighbor_entity = candidate_entity.split([self.cfg.entity_size,  self.cfg.entity_size * self.cfg.entity_neighbors], dim=-1)

#             cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
#             # cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

#             # cand_entity_emb = self.entity_encoder(candidate_entity, entity_mask).view(batch_size, -1, self.news_dim) # [8, 5, 400]
#         else:
#             cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

#         cand_final_emb = self.candidate_encoder(cand_title_emb, cand_title_emb, cand_origin_entity_emb)
#         # ----------------------------------------- Score ------------------------------------
#         score = self.click_predictor(cand_final_emb, user_emb)
#         loss = self.loss_fn(score, label)

#         return loss, score

#     def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask):
        
#         batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

#         # title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
#         # clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
#         clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

#         #--------------------Attention Pooling
#         if self.use_entity:
#             clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
#         else:
#             clicked_entity_emb = None
        
#         # clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)
#         clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_origin_emb, clicked_entity_emb)

#         user_emb = self.user_encoder(clicked_final_emb)  # [1, 400]

#         # ----------------------------------------- Candidate------------------------------------

#         if self.use_entity:
#             cand_entity_input = candidate_entity.unsqueeze(0)
#             entity_mask = entity_mask.unsqueeze(0)
#             origin_entity, neighbor_entity = cand_entity_input.split([self.cfg.entity_size,  self.cfg.entity_size * self.cfg.entity_neighbors], dim=-1)

#             cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
#             # cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

#         else:
#             cand_origin_entity_emb = None
#             cand_neighbor_entity_emb = None

#         cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), candidate_emb.unsqueeze(0), cand_origin_entity_emb)
#         # ---------------------------------------------------------------------------------------
#         # ----------------------------------------- Score ------------------------------------
#         scores = self.click_predictor(cand_final_emb, user_emb).view(-1).detach().cpu().tolist()

#         return scores
#         


class NAML(torch.nn.Module):
    def __init__(self, embedding_matrix, entity_emb, num_category, num_subcategory, cfg):
        super(NAML, self).__init__()
        self.news_dim = 400
        self.entity_dim = 100
        self.npratio = 4
        self.user_log_length = 50
        self.cfg = cfg
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)
        
        pretrain = torch.from_numpy(entity_emb).float()
        self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, 
                                                                   freeze=False, 
                                                                   padding_idx=0)
        
        self.news_encoder = NewsEncoder( word_embedding, num_category, num_subcategory)
        self.user_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
        self.candi_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
        self.user_encoder = UserEncoder()
        self.entity_encoder = EntityEncoder(cfg)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, args):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        history, history_mask, candidate, label = args
        
        if self.cfg.use_entity:
            e_his = history[:,:,-5:]
            history = history[:,:,:-5]
            e_candi = candidate[:,:,-5:]
            candidate = candidate[:,:,:-5]
        num_words = history.shape[-1]
        
        if self.cfg.use_entity:
            e_his = self.entity_embedding_layer(e_his)
            e_candi = self.entity_embedding_layer(e_candi)
            e_his = self.entity_encoder(e_his, None)
            e_candi = self.entity_encoder(e_candi, None)

            candidate = candidate.reshape(-1, num_words)
            candidate = self.news_encoder(candidate).reshape(-1, 1 + self.npratio, self.news_dim)
            candidate = self.candi_att(torch.stack([candidate, e_candi], dim=2).view(-1, 2, self.news_dim))
            candidate = candidate.view(-1, self.npratio+1, self.news_dim)

            history = history.reshape(-1, num_words)
            history = self.news_encoder(history).reshape(-1, self.user_log_length, self.news_dim)
            user_vec = self.user_att(torch.stack([history, e_his], dim=2).view(-1, 2, self.news_dim))
            user_vec = user_vec.view(-1, self.user_log_length, self.news_dim)
            
        else:
       
            candidate = candidate.reshape(-1, num_words)
            candidate = self.news_encoder(candidate).reshape(-1, 1 + self.npratio, self.news_dim)
            history = history.reshape(-1, num_words)
            user_vec = self.news_encoder(history).reshape(-1, self.user_log_length, self.news_dim)
        
        user_vec = self.user_encoder(user_vec, history_mask)
        score = torch.bmm(candidate, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
  
        return loss, score

