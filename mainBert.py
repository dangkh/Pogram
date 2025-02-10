import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from src.config import TrainConfig 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import BertModel
import shutil
import os
from tqdm import tqdm
from src.ultis import *
import nltk
nltk.download('punkt_tab')
cfg = TrainConfig
device: torch.device = torch.device(f"cuda:{cfg.deviceIndex}" if torch.cuda.is_available() else "cpu")


def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value

def read_news(news_path, mode='train'):
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}


    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, o_title, abstract, url, _, _ = splited
            update_dict(news_index, doc_id)

            title = o_title.lower()
            title = tokenizer(title, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
            tmp = title['input_ids'].sum().item()
            update_dict(news, doc_id, [title, category, subcategory])
            if mode == 'train':
                if use_category:
                    update_dict(category_dict, category)
                if use_subcategory:
                    update_dict(subcategory_dict, subcategory)


    if mode == 'train':
        return news, news_index, category_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def prepare_training_data(seed = 1009, npratio = 4):
    random.seed(seed)
    behaviors = []

    behavior_file_path = os.path.join(train_data_dir, 'behaviors.tsv')
    with open(behavior_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            iid, uid, time, history, imp = line.strip().split('\t')
            impressions = [x.split('-') for x in imp.split(' ')]
            pos, neg = [], []
            for news_ID, label in impressions:
                if label == '0':
                    neg.append(news_ID)
                elif label == '1':
                    pos.append(news_ID)
            if len(pos) == 0 or len(neg) == 0:
                continue
            for pos_id in pos:
                neg_candidate = get_sample(neg, npratio)
                neg_str = ' '.join(neg_candidate)
                new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                behaviors.append(new_line)

    random.shuffle(behaviors)
    processed_file_path = os.path.join(train_data_dir, f'behaviors_np{npratio}.tsv')
    with open(processed_file_path, 'w') as f:
        f.writelines(behaviors)
    return len(behaviors)


def get_doc_input(news, news_index, category_dict, subcategory_dict):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, num_words_title), dtype='int32')
    news_title_att = np.zeros((news_num, num_words_title), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32')
    news_subcategory = np.zeros((news_num, 1), dtype='int32')
    counter = 0
    for key in tqdm(news):
        title, category, subcategory = news[key]
        doc_index = news_index[key]
        news_title[doc_index] = title['input_ids']
        news_title_att[doc_index] = title['attention_mask']
        counter += 1
        news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return news_title, news_title_att, news_category, news_subcategory


class DatasetTrain(IterableDataset):
    def __init__(self, filename, news_index, news_combined):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_combined = news_combined
        self.user_log_length = 50
        self.npratio = 4

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_docs = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.user_log_length)
        user_feature = self.news_combined[click_docs]

        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.npratio)
        sample_news = neg[:label] + pos + neg[label:]
        news_feature = self.news_combined[sample_news]
        return user_feature, log_mask, news_feature, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class DatasetTest(DatasetTrain):
    def __init__(self, filename, news_index, news_scoring):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_scoring = news_scoring
        self.user_log_length = 50

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_docs = line[3].split()
        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.user_log_length)
        user_feature = self.news_scoring[click_docs]

        candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        news_feature = self.news_scoring[candidate_news]

        return user_feature, log_mask, news_feature, labels

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class AttentionPooling(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, emb_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, emb_dim
        """
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        '''
            Q: batch_size, n_head, candidate_num, d_k
            K: batch_size, n_head, candidate_num, d_k
            V: batch_size, n_head, candidate_num, d_v
            attn_mask: batch_size, n_head, candidate_num
            Return: batch_size, n_head, candidate_num, d_v
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)

        if attn_mask is not None:
            scores = scores * attn_mask.unsqueeze(dim=-2)

        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        '''
            Q: batch_size, candidate_num, d_model
            K: batch_size, candidate_num, d_model
            V: batch_size, candidate_num, d_model
            mask: batch_size, candidate_num
        '''
        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context = self.scaled_dot_product_attn(q_s, k_s, v_s, mask)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return output

class NewsEncoder(nn.Module):
    def __init__(self, num_category, num_subcategory):
        super(NewsEncoder, self).__init__()
        self.bertmodel = BertModel.from_pretrained("bert-base-uncased")
        for name, param in self.bertmodel.named_parameters():
            if not name.startswith("encoder.layer.11.output"):
                param.requires_grad = False
        news_dim = 400
        news_query_vector_dim = 200
        self.use_category = True
        self.use_subcategory = True
        category_emb_dim = 100
        self.num_words_title = 20
        self.category_emb = nn.Embedding(num_category + 1, category_emb_dim, padding_idx=0)
        self.category_dense = nn.Linear(category_emb_dim, news_dim)
        self.subcategory_emb = nn.Embedding(num_subcategory + 1, category_emb_dim, padding_idx=0)
        self.subcategory_dense = nn.Linear(category_emb_dim, news_dim)
        self.attn = AttentionPooling(768, news_query_vector_dim)
        self.ln = nn.Linear(768, news_dim)
        self.final_attn = AttentionPooling(news_dim, news_query_vector_dim)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        token = x[:, :num_words_title]
        attention_mask = x[:, num_words_title:num_words_title*2]
        outputs = self.bertmodel(input_ids=token, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state
        title_vecs = self.attn(outputs)
        title_vecs = self.ln(title_vecs)
        all_vecs = [title_vecs]

        start = self.num_words_title * 2
        if self.use_category:
            category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            category_vecs = self.category_dense(self.category_emb(category))
            all_vecs.append(category_vecs)
            start += 1
        if self.use_subcategory:
            subcategory = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            subcategory_vecs = self.subcategory_dense(self.subcategory_emb(subcategory))
            all_vecs.append(subcategory_vecs)

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)

            news_vecs = self.final_attn(all_vecs)
        return news_vecs


class NAML(torch.nn.Module):
    def __init__(self,num_category, num_subcategory, **kwargs):
        super(NAML, self).__init__()
        self.news_encoder = NewsEncoder(num_category, num_subcategory)
        self.user_encoder = UserEncoder()
        self.loss_fn = nn.CrossEntropyLoss()
        self.npratio = 4
        self.news_dim = 400
        self.user_log_length = 50

    def forward(self, history, history_mask, candidate, label):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        num_words = history.shape[-1]
        candidate_news = candidate.reshape(-1, num_words)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.npratio, self.news_dim)
        history_news = history.reshape(-1, num_words)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.user_log_length, self.news_dim)
        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score



class UserEncoder(nn.Module):
    def __init__(self):
        super(UserEncoder, self).__init__()
        news_dim = 400
        user_query_vector_dim = 200
        self.user_log_length = 50
        self.user_log_mask = False
        self.attn = AttentionPooling(news_dim, user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.user_log_mask:
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            user_vec = self.attn(news_vecs)
        return user_vec

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


set_random_seed(cfg.random_seed)

train_data_dir = './data/MINDsmall_train'
num_words_title = 20
use_category = True
use_subcategory = True
processed_file_path = os.path.join(train_data_dir, f'behaviors_np{4}.tsv')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
prepare_training_data()
# train():
news, news_index, category_dict, subcategory_dict = read_news(
    os.path.join(train_data_dir, 'news.tsv'), mode='train')
news_title, news_title_att, news_category, news_subcategory = get_doc_input(
    news, news_index, category_dict, subcategory_dict)
news_combined = np.concatenate([x for x in [news_title, news_title_att, news_category, news_subcategory] if x is not None], axis=-1)


dataset = DatasetTrain(processed_file_path, news_index, news_combined)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

model = NAML(len(category_dict), len(subcategory_dict))
optimizer = optim.Adam(model.parameters(), lr=0.0003)
model = model.to(device)
torch.set_grad_enabled(True)
model.train()

for ep in range(6):
    loss = 0.0
    accuary = 0.0
    print("EPOCH: " + str(ep))
    for cnt, (log_ids, log_mask, input_ids, targets) in tqdm(enumerate(dataloader)):
        log_ids = log_ids.to(device)
        log_mask = log_mask.to(device)
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        bz_loss, y_hat = model(log_ids, log_mask, input_ids, targets)
        loss += bz_loss.data.float()
        accuary += acc(targets, y_hat)
        optimizer.zero_grad()
        bz_loss.backward()
        optimizer.step()
    print(loss, accuary)
    checkpoint = {
			'epoch': ep,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,  # Save loss or any other metric
		}
    torch.save(checkpoint, f'./checkpoint/{ep}model.pth')

