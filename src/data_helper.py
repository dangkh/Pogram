import os
import random
import numpy as np
import torch
from typing import Callable
import logging
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
import collections
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_self_loops
import nltk
nltk.download('punkt')
    
import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import json
import itertools
from pathlib import Path

def update_dict(target_dict, key, value=None):
    """
    Function for updating dict with key / key+value

    Args:
        target_dict(dict): target dict
        key(string): target key
        value(Any, optional): if None, equals len(dict+1)
    """
    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value

def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)

def prepare_distributed_data(cfg, mode = "train"):
    # check
    target_file = os.path.join(cfg.data_dir + f'_{mode}', f"behaviors_np{cfg.npratio}_0.tsv")
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target_file is not exist. New behavior file in {target_file}')
    behaviors = []
    behavior_file_path = os.path.join(cfg.data_dir + f'_{mode}', 'behaviors.tsv')
    print(target_file, behavior_file_path)

    if mode == 'train':
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                iid, uid, time, history, imp = line.strip().split('\t')
                
                # his = history.split(' ')
                # # print(his, len(his))
                # if his[0] == '':
                #     continue

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
                    neg_candidate = get_sample(neg, cfg.npratio)
                    neg_str = ' '.join(neg_candidate)
                    new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                    behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)

    elif mode in ['val', 'test']:
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        behavior_file_path = os.path.join(cfg.data_dir + f'_{mode}', 'behaviors.tsv')
        with open(behavior_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                behaviors_per_file[i % cfg.gpu_num].append(line)

    print(f'[{mode}]Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(cfg.data_dir + f'_{mode}', f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)

def read_parsed_news(cfg, news, news_dict,
                     category_dict=None, subcategory_dict=None, entity_dict=None,
                     word_dict=None):
    news_num = len(news) + 1
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    news_entity = np.zeros((news_num, cfg.entity_size), dtype='int32')

    news_title = np.zeros((news_num, cfg.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]

        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]

        # entity
        entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
        news_entity[_news_index, :min(cfg.entity_size, len(_entity_ids))] = entity_index[:cfg.entity_size]

        for _word_id in range(min(cfg.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    # return news_title, news_entity, news_category, news_subcategory, news_index
    if cfg.use_entity:
        return news_title, news_category, news_subcategory, news_entity
    return news_title, news_category, news_subcategory


def read_raw_news(cfg, file_path, mode='train'):

    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir + '_test'}

    if mode in ['val', 'test']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:
        news = {}
        news_dict = {}
        entity_dict = {}
    
    if cfg.use_EnrichE:
        with open(cfg.data_dir + '_train/refine_data_full.json', 'r') as f:
            enrichedE = json.load(f)

    if cfg.genAbs:
        path = f"genAbs{cfg.absType}"
        with open(cfg.data_dir + f'_train/{path}.json', 'r') as f:
            genAbs = json.load(f)
            listGenKey = list(genAbs.keys())

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.

    num_line = len(open(file_path, encoding='utf-8').readlines())
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}]Processing raw news"):
            # split one line
            split_line = line.strip('\n').split('\t')
            news_id, category, subcategory, title, abstract, url, t_entity_str, _ = split_line
            if cfg.genAbs and news_id in listGenKey:
                title = genAbs[news_id][0]
            update_dict(target_dict=news_dict, key=news_id)

            # Entity
            if t_entity_str:
                entity_ids = [obj["WikidataId"] for obj in json.loads(t_entity_str)]
                if cfg.use_EnrichE and (news_id in enrichedE):
                    tmp = [x[1] for x in enrichedE[news_id]]
                    currentid = 0
                    while (len(entity_ids) < cfg.entity_size) and (currentid < len(tmp) - 1):
                        if tmp[currentid] not in entity_ids:
                            entity_ids.append(tmp[currentid])
                        currentid += 1
                [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
            else:
                entity_ids = t_entity_str

            
            tokens = word_tokenize(title.lower())

            update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                                news_dict[news_id]])

            if mode == 'train':
                update_dict(target_dict=category_dict, key=category)
                update_dict(target_dict=subcategory_dict, key=subcategory)
                word_cnt.update(tokens)

        if mode == 'train':
            word = [k for k, v in word_cnt.items() if v > 3]
            word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
            print(len(entity_dict))
            return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
        else:  # val, test
            print(len(entity_dict))
            return news, news_dict, None, None, entity_dict, None

def prepare_preprocess_bin(cfg, mode):
    if cfg.reprocess is True:
        # Glove
        newsPath = os.path.join(cfg.data_dir + f'_{mode}', 'news.tsv')
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict = read_raw_news(
            file_path=newsPath,
            cfg=cfg,
            mode=mode,
        )

        if mode == "train":
            pickle.dump(category_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "category_dict.bin"), "wb"))
            pickle.dump(subcategory_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "subcategory_dict.bin"), "wb"))
            pickle.dump(word_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "word_dict.bin"), "wb"))
        else:
            category_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "category_dict.bin"), "rb"))
            subcategory_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "subcategory_dict.bin"), "rb"))
            word_dict = pickle.load(open(os.path.join(cfg.data_dir + '_train', "word_dict.bin"), "rb"))

        pickle.dump(entity_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "entity_dict.bin"), "wb"))
        pickle.dump(nltk_news, open(os.path.join(cfg.data_dir + f'_{mode}', "nltk_news.bin"), "wb"))
        pickle.dump(nltk_news_dict, open(os.path.join(cfg.data_dir + f'_{mode}', "news_dict.bin"), "wb"))
        
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict,
                                              category_dict, subcategory_dict, entity_dict,
                                              word_dict)
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(news_input, open(os.path.join(cfg.data_dir + f'_{mode}', "nltk_token_news.bin"), "wb"))
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')    

def prepare_news_graph(cfg, mode='train'):
    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir + '_test'}

    nltk_target_path = Path(os.path.join(cfg.data_dir + f'_{mode}', "nltk_news_graph.pt"))

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return
    
    # -----------------------------------------News Graph------------------------------------------------
    behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    
    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        num_line = len(open(behavior_path, encoding='utf-8').readlines())
        with open(behavior_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"):
                line = line.strip().split('\t')

                # check duplicate user
                used_id = line[1]
                if used_id in user_set:
                    continue
                else:
                    user_set.add(used_id)

                # record cnt & read path
                history = line[3].split()
                if len(history) > 1:
                    long_edge = [news_dict[news_id] for news_id in history]
                    edge_list.append(long_edge)

        # edge count
        node_feat = nltk_token_news[:,:22] # keep only feature of title and category, exclude entity indices.
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):
            # Trajectory Graph
            if cfg.use_graph_type == 0:
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
                    # short_edges.append((edge[i + 1], edge[i]))
            elif cfg.use_graph_type == 1:
                # Co-occurence Graph fully connecting
                for i in range(len(edge) - 1):
                    for j in range(i+1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        edge_weights = Counter(short_edges)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        data = Data(x=torch.from_numpy(node_feat),
                edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)
    
        torch.save(data, target_path)
        print(data)
        print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
    
    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path, weights_only = False)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)
        
        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")

def prepare_neighbor_list(cfg, mode='train', target='news'):
    #--------------------------------Neighbors List-------------------------------------------
    print(f"[{mode}] Start to process neighbors list")

    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir}

    neighbor_dict_path = Path(data_dir[mode] ) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode] ) / f"{target}_weights_dict.bin"
    reprocess_flag = False
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] All {target} Neighbor dict exist !")
        return

    if target == 'news':
        target_graph_path = Path(data_dir[mode] ) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode] ) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path, weights_only = False)
    elif target == 'entity':
        target_graph_path = Path(data_dir[mode] ) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path, weights_only = False)
    else:
        assert False, f"[{mode}] Wrong target {target} "

    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr

    if cfg.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)
    
    # for each node (except 0)
    for i in range(1, len(target_dict)+1):
        dst_edges = torch.where(edge_index[1] == i)[0]          # i as dst
        neighbor_weights = edge_attr[dst_edges]
        neighbor_nodes = edge_index[0][dst_edges]               # neighbors as src
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
        neighbor_dict[i] = neighbor_nodes[indices].tolist()
        neighbor_weights_dict[i] = sorted_weights.tolist()
    
    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print(f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")


def prepare_entity_graph(cfg, mode='train'):
    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir}

    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path, weights_only = False)
        print("news_graph,", news_graph)
        entity_indices = news_graph.x[:, -8:-3].numpy()
        print("entity_indices, ", entity_indices.shape)

        entity_edge_index = []
        # -------- Inter-news -----------------
        # for entity_idx in entity_indices:
        #     entity_idx = entity_idx[entity_idx > 0]
        #     edges = list(itertools.combinations(entity_idx, r=2))
        #     entity_edge_index.extend(edges)

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)

        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        # --- Entity Graph Undirected
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
            
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path, weights_only = False)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(entity_dict) + 1),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(entity_dict) + 1)
        
        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")    

def prepare_preprocessed_data(cfg)  -> None:
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    # prepare_preprocess_bin(cfg, "test")

    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')
    # prepare_news_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    # prepare_neighbor_list(cfg, 'test', 'news')

    # prepare_entity_graph(cfg, 'train')
    # prepare_entity_graph(cfg, 'val')
    # # prepare_entity_graph(cfg, 'test')

    # prepare_neighbor_list(cfg, 'train', 'entity')
    # prepare_neighbor_list(cfg, 'val', 'entity')
    # # prepare_neighbor_list(cfg, 'test', 'entity')

    # # Entity vec process
    data_dir = {"train": cfg.data_dir + '_train', "val": cfg.data_dir + '_val', "test": cfg.data_dir}

    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    # test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    # test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    # Open the first file and read the contents
    with open(train_entity_emb_path, 'r') as file1:
        file1_content = file1.readlines()
    # Open the second file and read the contents
    with open(val_entity_emb_path, 'r') as file2:
        file2_content = file2.readlines()

    merged_content = file1_content + file2_content

    # Write the concatenated content to the output file
    with open(val_combined_path, 'w') as output_file:
        output_file.writelines(merged_content)