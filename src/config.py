from dataclasses import dataclass

@dataclass
class TrainConfig:
    random_seed: int = 42
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 16
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
    abs_size: int = 100
    entity_size: int = 5
    use_graph_type: int = 0 # adjacency connect or fully connect, fully set = 1
    directed: bool = False
    model_name: str = "Panel"
    use_entity: bool = True