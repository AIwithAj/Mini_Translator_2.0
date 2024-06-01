from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    raw_path:Path
    dataset_name:str
    raw_data:Path
    train:Path
    valid:Path
    test:Path
    



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list




@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_file: Path
    lang1:str
    lang2:str
    seq_len : int
    data_loader:Path
    batch_size: int



@dataclass(frozen=True)
class model_trainer_config:
    root_dir: Path
    seq_len:int
    num_epochs: int
    model_folder: str
    model_basename:str
    preload: None
    experiment_name: str
    lr:float
    d_model:int

    
    

@dataclass(frozen=True)
class model_eval_config:
    root_dir: Path



