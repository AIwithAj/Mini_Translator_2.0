from src.Mini_Translator_T.constants import *
from src.Mini_Translator_T.utils.common import read_yaml,create_directories
from src.Mini_Translator_T.entity import (DataIngestionConfig,DataValidationConfig,model_trainer_config,
                                          DataTransformationConfig,model_eval_config)


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
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
        

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params=self.params

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_file=config.tokenizer_file,
            lang1=params.lang1,
            lang2=params.lang2,
            seq_len=params.seq_len,
            data_loader=config.data_loader,
            batch_size=params.batch_size
        )

        return data_transformation_config
    

    def get_train_model_config(self) -> model_trainer_config:
        config = self.config.model_trainer
        params=self.params

        create_directories([config.root_dir])

        trainer_config=model_trainer_config(root_dir=config.root_dir,seq_len=params.seq_len,num_epochs=params.num_epochs,
                                            model_folder=config.model_folder,model_basename=config.model_basename,
                                            experiment_name=config.experiment_name,lr=params.lr,d_model=params.d_model,preload=None)
        
        return trainer_config
    
        
    def get_eval_model_config(self) -> model_eval_config:
        config = self.config.model_evaluation
        params=self.params

        create_directories([config.root_dir])

        eval_config=model_eval_config(root_dir=config.root_dir)
        
        return eval_config
        