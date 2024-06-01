import datasets
import json
from src.Mini_Translator_T.constants import *
from src.Mini_Translator_T.utils.common import read_yaml
from src.Mini_Translator_T.logging import logger
from torch.utils.data import  random_split


class DataIngestion:
    def __init__(self, config,params_filepath=PARAMS_FILE_PATH):
        self.config = config
        self.params=read_yaml(params_filepath)

    def convert_dataset_to_serializable(self, dataset):
        return [example for example in dataset]

    def initiate_data_ingestion(self):
        logger.info("Initiating dataIngestion..")
        try:
            ds_raw = datasets.load_dataset(self.config.dataset_name, f'{self.params.lang1}-{self.params.lang2}', split = 'train')
            
            serializable_dataset = {
                "train": list(ds_raw)[1:],
            }

        # Save the serializable dataset to a JSON file
            with open(self.config.raw_data, 'w') as json_file:
                json.dump(serializable_dataset, json_file, indent=4)
        except Exception as e:
            logger.info("incorrect dataset")
            raise e
        
            # Splitting the dataset for training and validation
        train_ds_size = int(0.9 * len(ds_raw)) # 90% for training
        val_ds_size = len(ds_raw) - train_ds_size # 10% for validation
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # Randomly splitting the dataset


        train_data_serializable = self.convert_dataset_to_serializable(train_ds_raw)
        valid_data_serializable = self.convert_dataset_to_serializable(val_ds_raw)

        logger.info("saving train, valid..")
        with open(self.config.train, 'w') as train_file:
            json.dump(train_data_serializable, train_file, indent=4)
        with open(self.config.valid, 'w') as valid_file:
            json.dump(valid_data_serializable, valid_file, indent=4)


        logger.info("data_ingestion successfully saved")
