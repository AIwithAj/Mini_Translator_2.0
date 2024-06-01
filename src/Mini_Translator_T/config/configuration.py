from src.Mini_Translator_T.constants import *
from src.Mini_Translator_T.utils.common import read_yaml,create_directories
from src.Mini_Translator_T.entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) ->DataIngestionConfig:
        config=self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.raw_path])


        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            raw_path=config.raw_path,
            dataset_name=config.dataset_name,
            raw_data=config.data_files.raw_data,
            train=config.data_files.train,
            valid=config.data_files.validation,
            test=config.data_files.test
            
        )

        return data_ingestion_config
        
        