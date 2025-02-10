from dataclasses import dataclass
from typing import Any, Dict

@dataclass

class TrainConfig:
    random_seed: int = 1009
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 128
    gradient_accumulation_steps: int = 8  # batch_size = 16 x 8 = 128
    epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_len: int = 100
    reprocess: bool = False
    reprocess_neighbors : bool = False
    data_dir: str = "./data/MINDsmall"
    gpu_num: int = 1
    title_size: int = 20
    abstract_size: int = 50
    entity_size: int = 5
    use_graph_type: int = 0 # adjacency connect or fully connect, fully set = 1
    directed: bool = False
    model_name: str = "GLORY"
    entity_global: bool = False
    glove_path: str = './data/glove.840B.300d.txt'
    word_emb_dim : int = 300
    head_num: int = 4
    head_dim: int = 100
    entity_emb_dim : int = 100
    entity_neighbors : int = 10
    attention_hidden_dim: int = 200
    dropout_probability: float = 0.2
    his_size : int = 50
    k_hops: int = 1
    num_neighbors : int = 4
    use_graph : bool = False
    use_entity : bool = False
    use_EnrichE: bool = False
    early_stop_patience: int = 5
    prototype: bool = True
    genAbs : bool = False 
    absType: int = 0  # 0: direct; 1: via entity
    deviceIndex : int = 0

    def update(self, args: Dict[str, Any]):
        if isinstance(args, dict):
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            for key, value in vars(args).items():
                if hasattr(self, key):
                    setattr(self, key, value)