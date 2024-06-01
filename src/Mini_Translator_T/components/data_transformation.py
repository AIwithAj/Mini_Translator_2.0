
from src.Mini_Translator_T.config.configuration import DataTransformationConfig
from src.Mini_Translator_T.constants import *
from src.Mini_Translator_T.utils.common import read_yaml,casual_mask


import os
from src.Mini_Translator_T.logging import logger
import json


#HuggingFace linraries
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
import torch
from src.Mini_Translator_T.logging import logger  # Make sure to import the logger
from typing import Any

class BilingualDataset(torch.utils.data.Dataset):

    # This takes in the dataset contaning sentence pairs, the tokenizers for target and source languages, and the strings of source and target languages
    # 'seq_len' defines the sequence length for both languages
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Defining special tokens by using the target language tokenizer
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    # Total number of instances in the dataset (some pairs are larger than others)
    def __len__(self):
        return len(self.ds)

    # Using the index to retrive source and target texts
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]

        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenizing source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Computing how many padding tokens need to be added to the tokenized texts
        # Source tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # Subtracting the two '[EOS]' and '[SOS]' special tokens
        # Target tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Subtracting the '[SOS]' special token

        # If the texts exceed the 'seq_len' allowed, it will raise an error. This means that one of the sentences in the pair is too long to be processed
        # given the current sequence length limit (this will be defined in the config dictionary below)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Building the encoder input tensor by combining several elements
        encoder_input = torch.cat(
            [
            self.sos_token, # inserting the '[SOS]' token
            torch.tensor(enc_input_tokens, dtype = torch.int64), # Inserting the tokenized source text
            self.eos_token, # Inserting the '[EOS]' token
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64) # Addind padding tokens
            ]
        )

        # Building the decoder input tensor by combining several elements
        decoder_input = torch.cat(
            [
                self.sos_token, # inserting the '[SOS]' token
                torch.tensor(dec_input_tokens, dtype = torch.int64), # Inserting the tokenized target text
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # Addind padding tokens
            ]

        )

        # Creating a label tensor, the expected output for training the model
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64), # Inserting the tokenized target text
                self.eos_token, # Inserting the '[EOS]' token
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64) # Adding padding tokens

            ]
        )

        # Ensuring that the length of each tensor above is equal to the defined 'seq_len'
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, config_filepath=CONFIG_FILE_PATH):
        self.config = config
        
        self.config2 = read_yaml(config_filepath)

        # Defining Tokenizer

    # Iterating through dataset to extract the original sentence and its translation
    def get_all_sentences(self,ds, lang):
        for pair in ds:
            yield pair['translation'][lang]

    def build_tokenizer(self,config, ds, lang):

        # Crating a file path for the tokenizer
        tokenizer_path = Path(config.format(lang))

        # Checking if Tokenizer already exists
        if not Path.exists(tokenizer_path):

            # If it doesn't exist, we create a new one
            tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]')) # Initializing a new world-level tokenizer
            tokenizer.pre_tokenizer = Whitespace() # We will split the text into tokens based on whitespace

            # Creating a trainer for the new tokenizer
            trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]",
                                                        "[SOS]", "[EOS]"], min_frequency = 2) # Defining Word Level strategy and special tokens

            # Training new tokenizer on sentences from the dataset and language specified
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer = trainer)
            tokenizer.save(str(tokenizer_path)) # Saving trained tokenizer to the file path specified at the beginning of the function
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path)) # If the tokenizer already exist, we load it
        return tokenizer # Returns the loaded tokenizer or the trained tokenizer
    
    


    def initiate_tokenization(self):
        ingestion_config = self.config2.data_ingestion

        with open(ingestion_config.data_files.raw_data, 'r') as json_file:
            ds_raw = json.load(json_file)["train"]
            ds_raw=Dataset.from_pandas(pd.DataFrame(ds_raw))

        tokenizer_src = self.build_tokenizer(self.config.tokenizer_file, ds_raw, self.config.lang1)
        tokenizer_tgt = self.build_tokenizer(self.config.tokenizer_file, ds_raw, self.config.lang2)

        logger.info(['source tokenizerand target tokenizer saved succefully'])

        with open(ingestion_config.data_files.train, 'r') as json_file:
            train_ds_raw = json.load(json_file)

        with open(ingestion_config.data_files.validation, 'r') as json_file:
            val_ds_raw = json.load(json_file)



        # Convert the loaded lists of dictionaries to datasets.Dataset objects
        train_ds_raw = Dataset.from_pandas(pd.DataFrame(train_ds_raw))
        val_ds_raw = Dataset.from_pandas(pd.DataFrame(val_ds_raw))


       # Processing data with the BilingualDataset class, which we will define below
        train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, self.config.lang1, self.config.lang2, self.config.seq_len)
        val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, self.config.lang1, self.config.lang2, self.config.seq_len)


        
        # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and target languages
        max_len_src = 0
        max_len_tgt = 0
        for pair in ds_raw:
                src_ids = tokenizer_src.encode(pair['translation'][self.config.lang1]).ids
                tgt_ids = tokenizer_src.encode(pair['translation'][self.config.lang2]).ids
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))
        print(src_ids)
        logger.info(f'Max length of source sentence: {max_len_src}')
        logger.info(f'Max length of target sentence: {max_len_tgt}')
        

            # Creating dataloaders for the training and validadion sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
        train_dataloader = DataLoader(train_ds, batch_size = self.config.batch_size, shuffle = True) # Batch size will be defined in the config dictionary
        val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

        logger.info(f"Length of train data loader: {len(train_dataloader)}")
        logger.info(f"Length of valid data loader: {len(val_dataloader)}")

        # Saving dataloaders as list of batches
        root_dir=self.config.root_dir
        torch.save(list(train_dataloader), os.path.join(root_dir, "train_data_loader.pth"))  # Changed line
        torch.save(list(val_dataloader), os.path.join(root_dir, "valid_data_loader.pth"))  # Changed line

        logger.info(f"Data loaders saved to {root_dir}")
        logger.info("Data transformation successfully completed")

        # return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt # Returning the DataLoader objects and tokenizers
 


       