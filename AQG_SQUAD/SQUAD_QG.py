#%%
import pandas as pd
import numpy as np
import torch
import os
import sys
import nltk
import re, string, timeit

import pytorch_lightning as pl
from typing import List, Dict, Union
from torch.nn import MultiLabelSoftMarginLoss
from torch.optim import Optimizer

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,T5Model,
    T5TokenizerFast as T5Tokenizer
)

from transformers.modeling_outputs import SequenceClassifierOutputWithPast

print(torch.Tensor(5))

WORKING_DIR : str = os.path.join(os.getcwd(), "AQG_SQUAD")

TRAINING_PATH_DATASET : str = os.path.join(WORKING_DIR, "SQUAD_dataset", "UsedDataset", "train_squad_df.csv")
VALIDATING_PATH_DATASET : str = os.path.join(WORKING_DIR, "SQUAD_dataset", "UsedDataset", "val_squad_df.csv")

TRAINING_DF : pd.DataFrame = pd.read_csv(TRAINING_PATH_DATASET, delimiter=",").drop("Unnamed: 0", axis=1)
VALIDATING_DF : pd.DataFrame = pd.read_csv(VALIDATING_PATH_DATASET, delimiter=",").drop("Unnamed: 0", axis=1)
print(f"Train_Df Shape : {TRAINING_DF.shape})")
print(f"Val_df Shape : {VALIDATING_DF.shape})")
#%%

TITLE : str ='title'
PARAGRAPHS : str = 'paragraphs'
CONTEXT : str = "context"
QAS : str = "qas"
QUESTION : str ="question"
ANSWERS : str ="answers"
ANSWER_START : str ="answer_start"
ANSWER_TEXT : str = "text"
ANSWER_END : str = "answer_end"
CONTEXT_SENTENCE : str = "context_sentence"
print(TRAINING_DF.isna().sum())
#%%
def SplitDatasetNQG(trainDf : pd.DataFrame,
                    valDf : pd.DataFrame,
                    columnsToDrop : List[str] = [CONTEXT_SENTENCE, ANSWER_START, ANSWER_END], 
                    ):
    
    dfCopy : pd.DataFrame =  trainDf.copy()
    dfCopy.dropna(inplace= True)
    
    dfCopy.rename(columns = {ANSWER_TEXT : 'answer_text'})
    
    dfCopy.drop(columns=columnsToDrop, inplace=True, axis=1)
    
    test_df : pd.DataFrame = dfCopy[:11877]
    train_df : pd.DataFrame = dfCopy[11877:]
    
    devCopy : pd.DataFrame = valDf.copy()
    devCopy.drop(columns=columnsToDrop, inplace=True, axis=1)
    
    print(f"Train Shape : {train_df.shape}")
    print(f"Val Shape : {devCopy.shape}")
    print(f"Test Shape : {test_df.shape}")

    return train_df, devCopy, test_df
    
Training_data, Validating_data, Testing_data  = SplitDatasetNQG(TRAINING_DF, VALIDATING_DF)

#%%
#%%
MASKING_CHANCE : float = 0.3
SEP_TOKEN : str = ":"
MAX_LENGTH_STR  : str = "max_length"
DEFAULT_TENSORS : str = "pt"

class QGDataset(Dataset):
    def __init__(self, tokenizer : T5Tokenizer, data : pd.DataFrame,
                 source_max_token_len, target_max_token_len ):
        self.tokenizer : T5Tokenizer = tokenizer
        self.data : pd.DataFrame = data 
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
    
    def __len__(self):
      return len(self.data)
  
    def __getitem__(self, index):
        current_data = self.data.iloc[index,:]
        answer : str = "[MASK]"
        if( np.random.rand() > MASKING_CHANCE ):
            answer = current_data["answer_text"]
            
        source_encoding = self.tokenizer(
            "{} {} {}".format( answer, SEP_TOKEN, current_data[CONTEXT]),
            padding=MAX_LENGTH_STR,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors= DEFAULT_TENSORS,
        )
        
        target_encoding = self.tokenizer(
            "{} {} {}".format(current_data["answer_text"], SEP_TOKEN, current_data[QUESTION]),
            max_length=self.target_max_token_len,
            padding=MAX_LENGTH_STR,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors=DEFAULT_TENSORS,
        )
        
        # input_ids For target sequence
        labels = target_encoding['input_ids']
        
        # Ignore the padding !
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return dict(
            answer_text = current_data[ANSWER_TEXT],
            context = current_data[CONTEXT],
            question = current_data[QUESTION],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )
MODEL_NAME : str = "t5-small"
SOURCE_MAX_TOKEN : int = 300
TARGET_MAX_TOKEN : int = 80 
N_EPOCHS : int = 5
BATCH_SIZE : int = 16
LEARNING_RATE : float = 0.0001

TOKENIZER : T5Tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.add_tokens(SEP_TOKEN)

TOKENIZER_LEN : int = len(TOKENIZER)

class QGDataLoader(DataLoader):
    
    def __init__(self,
                 training_data : Dataset,
                 validating_data : Dataset,
                 testing_data : Dataset,
                 ):
        super().__init__()
        self.training_data  : Dataset = training_data
        self.validating_data : Dataset = validating_data
        self.testing_data : Dataset = testing_data
        
        self.trainingDataLoader : DataLoader 
        self.validatingDataLoader : DataLoader
        self.testingDataLoader :DataLoader

    def setupLoader(self):
        self.trainingDataLoader = DataLoader(self.training_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             )
        self.validatingDataLoader : DataLoader = DataLoader( self.validating_data, 
                                                            batch_size=1,
                                                            )
        
        self.testingDataLoader : DataLoader = DataLoader(self.testing_data,
                                                         batch_size=1)
    
class QGModel(torch.nn.Module):
    def __init__(self):
        super(QGModel, self).__init__()
        self.model : T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)
        self.model.resize_token_embeddings(TOKENIZER_LEN)
    def forward(self, 
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                labels : torch.Tensor = None
                ):
        output : SequenceClassifierOutputWithPast = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

class TrainerQGVanilla():
    
    def __init__(self, model : QGModel, 
                 optimizer : AdamW,
                 training_dataLoader : DataLoader,
                 validating_dataLoader : DataLoader,
                 testing_dataLoader : DataLoader
                 ):
        
        self.model : QGModel = model
        self.optimizer : Optimizer = optimizer
        self.trainingDataLoader : DataLoader = training_dataLoader
        self.validatingDataLoader : DataLoader = validating_dataLoader
        self.testingDataLoader : DataLoader = testing_dataLoader
    
    def training_step(self, trainBatch : torch.Tensor):
        self.model.train()
        total_Loss : torch.Tensor = torch.Tensor(0)
        
        for data in trainBatch:
            input_ids : torch.Tensor = data['input_ids']
            attention_mask : torch.Tensor = data['attention_mask']
            labels : torch.Tensor = data['labels']
            self.optimizer.zero_grad()
            outputs : SequenceClassifierOutputWithPast = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            self.optimizer.step()
            total_Loss += outputs.loss.item()
        return total_Loss / len(trainBatch)
    
    def validation_step(self, data : torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            input_ids : torch.Tensor = data['input_ids']
            attention_mask : torch.Tensor = data['attention_mask']
            labels : torch.Tensor = data['labels']
            output : SequenceClassifierOutputWithPast = self.model(input_ids = input_ids, attention_mask=attention_mask, labels=labels ) # Only 1 batch at a time
            return output.loss

    def test_step(self, data : torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            input_ids : torch.Tensor = data['input_ids']
            attention_mask : torch.Tensor = data['attention_mask']
            labels : torch.Tensor = data['labels'] 
            output : SequenceClassifierOutputWithPast = self.model(input_ids = input_ids, attention_mask=attention_mask, labels=labels )
            return output.loss
    
    def trainModel(self):
        for epoch in range(N_EPOCHS):
            train_loss : torch.Tensor = self.training_step(self.trainingDataLoader)
            print(f"Epoch : {epoch + 1}, Train Loss: {train_loss}")
            
            val_loss : torch.Tensor = sum(self.validation_step(data) for data in self.validatingDataLoader) / len(self.validatingDataLoader)
            print(f"Epoch : {epoch + 1}, Val Loss: {train_loss}")

            test_loss : torch.Tensor = sum(self.test_step(data) for data in self.testingDataLoader) / len (self.testingDataLoader)        
            print(f"Epoch : {epoch + 1}, Test Loss: {train_loss}")

class QGDataLoader_PL(pl.LightningDataModule):
    def __init__(self, tokenizer : T5Tokenizer, 
                 train : pd.DataFrame, valid: pd.DataFrame,
                 test : pd.DataFrame
                 ):
        self.tokenizer : T5Tokenizer = tokenizer
        self.train : pd.DataFrame = train
        self.valid : pd.DataFrame = valid 
        self.test : pd.DataFrame = test
        
        self.trainingDataset : Dataset
        self.validatingDataset : Dataset
        self.testingDataset : Dataset

    def setup(self):
        # Do the ETL Here
        self.trainingDataset  = QGDataset(self.tokenizer, self.train,
                                                  SOURCE_MAX_TOKEN, TARGET_MAX_TOKEN) 
        self.validatingDataset = QGDataset(self.tokenizer, self.valid,
                                                     SOURCE_MAX_TOKEN, TARGET_MAX_TOKEN)
        self.testingDataset  = QGDataset(self.tokenizer, self.test,
                                                  SOURCE_MAX_TOKEN, TARGET_MAX_TOKEN)
    def train_dataLoader(self):
        return DataLoader(self.trainingDataset, shuffle=True, batch_size=BATCH_SIZE, )

    def val_dataloader(self):
        return DataLoader(self.validatingDataset, batch_size= 1 )

    def test_dataloader(self):
        return DataLoader(self.testingDataset, batch_size=1)

class QGModel_PL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model : T5ForConditionalGeneration =  T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(TOKENIZER_LEN)
    
    def forward(self, input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                labels : torch.Tensor=None):
        output : SequenceClassifierOutputWithPast = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
        return output.loss, output.logits
    
    def training_step(self, batch : Dict[str, torch.Tensor] , batch_idx):
        input_ids : torch.Tensor = batch['input_ids']
        attention_mask : torch.Tensor = batch['attention_mask']
        labels : torch.Tensor = batch['labels']
        
        loss, output_logits = self(input_ids, attention_mask, labels) # Call the forward
        self.log('train_loss: ', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch : torch.Tensor, batch_idx):
        input_ids : torch.Tensor = batch['input_ids']
        attention_mask : torch.Tensor = batch['attention_mask']
        labels : torch.Tensor = batch['labels']

        loss, _ = self(input_ids, attention_mask, labels)

        self.log("validation_loss : ", loss, prog_bar=True, Logger=True)
        return loss
    
    def test_step(self, batch : torch.Tensor, batch_idx):
        
        input_ids : torch.Tensor = batch['input_ids']
        attention_mask : torch.Tensor = batch['attention_mask']
        labels : torch.Tensor = batch['labels']

        loss, _ = self(input_ids, attention_mask, labels)

        self.log("Testing_Loss: ", loss, prog_bar=True, Logger=True)
        return loss
        
        
        