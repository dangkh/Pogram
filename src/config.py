from dataclasses import dataclass

@dataclass
class TrainConfig:
    random_seed: int = 42
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 32
    gradient_accumulation_steps: int = 8  # batch_size = 16 x 8 = 128
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_len: int = 100
    reprocess: bool = False
    reprocess_neighbors : bool = False
    data_dir: str = "./data/MINDsmall"
    gpu_num: int = 1
    title_size: int = 30
    abstract_size: int = 50
    entity_size: int = 5
    use_graph_type: int = 0 # adjacency connect or fully connect, fully set = 1
    directed: bool = False
    model_name: str = "GLORY"
    use_entity: bool = True
    glove_path = './data/glove.840B.300d.txt'
    word_emb_dim = 300
    head_num = 20
    head_dim = 20
    entity_emb_dim = 100
    entity_neighbors = 10
    attention_hidden_dim = 200
    dropout_probability = 0.2
    his_size = 50
    k_hops = 2
    num_neighbors = 8
    use_graph = True
    early_stop_patience = 5
