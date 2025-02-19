import torch
import torch.nn as nn
from torch.nn import ReLU
from torch_geometric.nn import Sequential, GatedGraphConv, GCNConv,GATv2Conv, global_mean_pool, LayerNorm

from .base.layers import *
from .component.user_encoder import UserEncoder
from .component.entity_encoder import EntityEncoder
from .component.news_encoder import *


class NAML(torch.nn.Module):
    def __init__(self, embedding_matrix, entity_emb, num_category, num_subcategory, cfg):
        super(NAML, self).__init__()
        self.news_dim = 400
        self.entity_dim = 100
        self.npratio = 4
        self.user_log_length = cfg.his_size
        self.cfg = cfg
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=False,
                                                      padding_idx=0)
        
        self.news_encoder = NewsEncoder( word_embedding, num_category, num_subcategory, cfg.title_size)
        self.user_encoder = UserEncoder(cfg)
        
        if cfg.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, 
                                                                       freeze=False, 
                                                                       padding_idx=0)
            self.user_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
            self.candi_att = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)
            self.entity_encoder = EntityEncoder(cfg)

        if cfg.use_graph:
            self.g_hidden_dim = 64
            self.gnn1 = GCNConv(self.news_dim, self.g_hidden_dim)
            self.gnn2 = GCNConv(self.g_hidden_dim, self.g_hidden_dim)
            self.gnn3 = GCNConv(self.g_hidden_dim, self.g_hidden_dim)
            self.gln = nn.Linear(self.g_hidden_dim, self.news_dim)
            self.loc_glob_att = MultiHeadSelfAttention(self.news_dim, cfg.head_num , cfg.head_dim , cfg.head_dim)
            self.graph2newsDim = nn.Linear(cfg.head_dim*cfg.head_num, self.news_dim)
            self.loc_glob_att2 = AttentionPooling(self.news_dim, 128)
            self.glob_mean = global_mean_pool
            self.relu = nn.LeakyReLU(0.2)
            self.gnorm = nn.LayerNorm(self.g_hidden_dim)
            self.user_attg = AttentionPooling(self.news_dim, cfg.attention_hidden_dim)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, args):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        if self.cfg.use_graph:
            graph_batch, mapId, history, history_mask, candidate, label = args
            graph_vec, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
            graph_vec = self.news_encoder(graph_vec)
            graph_vec = self.gnn1(graph_vec, edge_index)
            graph_vec = self.relu(graph_vec)
            graph_vec = self.gnn2(graph_vec, edge_index)
            graph_vec = self.relu(graph_vec)
            # graph_vec = self.gnn3(graph_vec, edge_index)
            # graph_vec = self.relu(graph_vec)
            # graph_vec = self.gnorm(graph_vec)
            graph_vec = self.gln(graph_vec)
            graph_vec = self.relu(graph_vec)
            graphBYuser = graph_vec[mapId.view(-1)]
            graphBYuser = graphBYuser.view(-1, self.user_log_length, self.news_dim)
            graph_vec = self.glob_mean(graph_vec, batch)
        else:
            _, _, history, history_mask, candidate, label = args
        
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
        history = history.reshape(-1, num_words)
        user_vec = self.news_encoder(history).reshape(-1, self.user_log_length, self.news_dim)
        
        if self.cfg.use_entity:
            candidate = self.candi_att(torch.stack([candidate, e_candi], dim=2).view(-1, 2, self.news_dim))
            candidate = candidate.view(-1, self.npratio+1, self.news_dim)
            user_vec = self.user_att(torch.stack([user_vec, e_his], dim=2).view(-1, 2, self.news_dim))
            user_vec = user_vec.view(-1, self.user_log_length, self.news_dim)
        
        if self.cfg.use_graph:
            user_vec = self.user_attg(torch.stack([user_vec, graphBYuser], dim=2).view(-1, 2, self.news_dim))
            user_vec = user_vec.view(-1, self.user_log_length, self.news_dim)
            
        user_vec = self.user_encoder(user_vec, history_mask)
        # if self.cfg.use_graph:
        #     graph_vec = self.loc_glob_att(graph_vec, user_vec, user_vec)
        #     graph_vec = self.graph2newsDim(graph_vec).view(-1, self.news_dim)
        #     graph_vec = self.relu(graph_vec)
        #     user_vec = torch.stack([user_vec, graph_vec], dim=1)
        #     user_vec = self.loc_glob_att2(user_vec)
        score = torch.bmm(candidate, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
  
        return loss, score
