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
