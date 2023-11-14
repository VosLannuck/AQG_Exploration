#%%

import json
import pandas as pd
import numpy as np
import torch
import sys
import os
import nltk
import re, string, timeit

from typing import List,Dict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
#sys.path.append(os.getcwd()+"/AQG_SQUAD")
print(sys.path)
import pytorch_lightning as pl
SEED : int = 42
torch.random.manual_seed(SEED)
#pl.seed_everything(SEED)


#%%


WORKING_DIR : str = os.getcwd() + "/AQG_SQUAD"

SQUAD_TRAIN_PATH : str = os.path.join(WORKING_DIR, "SQUAD_dataset/train.json")
SQUAD_DEV_PATH : str = os.path.join(WORKING_DIR,"SQUAD_dataset/dev.json")

SQUAD_TRAIN_DF : pd.DataFrame = pd.read_json(SQUAD_TRAIN_PATH)
SQUAD_DEV_DF : pd.DataFrame = pd.read_json(SQUAD_DEV_PATH)
DATA_COLUMN : str = "data"

TITLE : str ='title'
PARAGRAPHS : str = 'paragraphs'
CONTEXT : str = "context"
QAS : str = "qas"
QUESTION : str ="question"
ANSWERS : str ="answers"
ANSWER_START : str ="answer_start"
ANSWER_TEXT : str = "text"
ANSWER_END : str = "answer_end"
EXCLUDE_PUNCTUATIONS : List[str] = set(string.punctuation)
REGEX = re.compile('[%s]' % re.escape(string.punctuation))

CONTEXT_SENTENCE : str = "context_sentence"
TARGET_COLUMNS : List[str] = [QUESTION, CONTEXT, CONTEXT_SENTENCE, ANSWER_TEXT, ANSWER_START, ANSWER_END ]

#%%
def showQuestion(df : pd.DataFrame ,titleId : int = 0, paragraphId : int = 0, questionId : int = 0) -> None:
    
    data : pd.Series = df.loc[titleId, DATA_COLUMN]
    
    title : str = data[TITLE]
    paragraph : str = data[PARAGRAPHS][paragraphId][CONTEXT]
    question : str = data[PARAGRAPHS][paragraphId][QAS][questionId][QUESTION]
    answer : str = data[PARAGRAPHS][paragraphId][QAS][questionId][ANSWERS][0][ANSWER_TEXT]
    answerStart : str = data[PARAGRAPHS][paragraphId][QAS][questionId][ANSWERS][0][ANSWER_START]

    
    
    print("--Title--")
    print(title)
    print("---------")
    
    print("--Paragraph--")
    print(paragraph)
    print("---------")
    
    print("--Question--")
    print(question)
    print("---------")
    
    print("--Answer--")
    print(answer)
    print("Answer start :  %s" %(answerStart) )
    print("---------")
 
   
def CheckDataset(df : pd.DataFrame):
    titleCounts : int = len(df[DATA_COLUMN])
    totalparagraph : int = 0
    totalQuestions : int = 0
    
    for title in range(titleCounts):
        data : pd.Series = df[DATA_COLUMN][title]
        paragraphsCount : int = len(data[PARAGRAPHS])
        totalparagraph += paragraphsCount

        for paragraphId in range(paragraphsCount):
            totalQuestions += len(data[PARAGRAPHS][paragraphId][QAS])
    
    print("TitleCounts : %s " % (titleCounts))
    print("Titles : %s " % (totalparagraph))
    print("Questions : %s " % (totalQuestions))
#%%
def PreprocessText(text : str) -> str:
    text = text.lower()
    text = REGEX.sub('', text)
    return text
    
def MakeDatasetPerQuestion(df : pd.DataFrame, save_csv_name="train_squad_df.csv"):
    df_target : pd.DataFrame = pd.DataFrame(columns=TARGET_COLUMNS)
    
    titlesCount : int = len(df[DATA_COLUMN])
    data : pd.Series = df[DATA_COLUMN]
    for titleId in range(titlesCount):
        new_row : Dict = {}
        title : str = data[titleId]
        paragraphs : List = data[titleId][PARAGRAPHS]
        totalParagraphs : int = len(paragraphs)
        for paragraphId in range(totalParagraphs):  
            context : str = paragraphs[paragraphId][CONTEXT]
            
            qas : str = paragraphs[paragraphId][QAS]
            totalQuestionAnswerPairs : int = len(qas)
            
            for pairId in range(totalQuestionAnswerPairs):
                answer : str = qas[pairId][ANSWERS][0][ANSWER_TEXT]
                answer_start : int = qas[pairId][ANSWERS][0][ANSWER_START]
                answer_end : int = answer_start + len(answer)
                question : str = qas[pairId][QUESTION]
                context_sentence : str = "<EMPTY>"
                
                sentences = nltk.tokenize.sent_tokenize(context)
            
                sentenceStart = 0
                for sentence in sentences:
                    if (sentenceStart + len(sentence) >= answer_start):
                        context_sentence = sentence
                        break
                    sentenceStart += len(sentence) + 1
                    
                new_row = {
                    QUESTION : question,
                    CONTEXT : context,
                    CONTEXT_SENTENCE : context_sentence,
                    ANSWER_TEXT : answer,
                    ANSWER_START : answer_start,
                    ANSWER_END : answer_end
                }
                df_target = pd.concat([ df_target, pd.DataFrame([new_row])], ignore_index=True)
    df_target.to_csv(save_csv_name)
    print("The Dataset has been saved :D")
                
                
#MakeDatasetPerQuestion(SQUAD_TRAIN_DF)
MakeDatasetPerQuestion(SQUAD_DEV_DF,"val_squad_df.csv")
#%%
#%%
if __name__ == "__main__":
    showQuestion(SQUAD_TRAIN_DF)
    #CheckDataset(SQUAD_TRAIN_DF)