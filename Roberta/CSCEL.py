#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import random
import re
import warnings
import math

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tqdm.notebook import tqdm
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, BertModel

import category_encoders as ce
import nlp
import xfeat
from xfeat import TargetEncoder

warnings.filterwarnings("ignore")

INPUT = "./input"
FOLDS = 5  # kfoldの数
SEED = 42


# In[2]:


train = pd.read_csv(os.path.join(INPUT, "Reviews.csv"))[['Score', 'Text']].sample(n=10000, random_state=SEED)
train["Score"] = train["Score"]-1


# In[3]:


train_y = train["Score"].copy()
train_X = train['Text'].copy()


# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)


# In[6]:


skfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
cv = list(skfold.split(train_X, train_y))


# In[7]:


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "./models_bert/"
MODEL_NAME = 'roberta-base'
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 10
NUM_SPLITS = 5


# In[8]:


def CIQ(label_num):
    return [[-math.log2((label_num[i]/2+sum(label_num[i+1:j+1]))/sum(label_num)) if i<=j 
             else -math.log2((label_num[i]/2+sum(label_num[j:i]))/sum(label_num))
             for j in range(len(label_num))] for i in range(len(label_num))]

def CEM(all_labels, all_preds):
    prox = CIQ(np.bincount(all_labels))
    return sum([prox[pred][label] for label, pred in zip(all_labels, all_preds)])/sum([prox[label][label] for label in all_labels])

def calc_PCNL(labels_count):
    return [(sum(labels_count[0:i])+sum(labels_count[1:i+1]))/(2*sum(labels_count)-labels_count[0]-labels_count[len(labels_count)-1])
           for i in range(len(labels_count))] 

def calc_OWAS_worst(all_labels, prox):
    labels_count = np.bincount(all_labels)
    return sum([max(prox[label],1-prox[label]) /labels_count[label]for label in all_labels])

def calc_ROWSS_worst(all_labels, prox):
    labels_count = np.bincount(all_labels)
    return sum([max(prox[label],1-prox[label])**2 /labels_count[label]for label in all_labels])

def OWAS_from_pred(all_labels, all_preds):
    labels_count = np.bincount(all_labels)
    prox = calc_PCNL(labels_count)
    OWE_worst = calc_OWAS_worst(all_labels, prox)
    return 1-sum([abs(prox[pred]-prox[label])/labels_count[label] for label, pred in zip(all_labels, all_preds)])/OWE_worst

def ROWSS_from_pred(all_labels, all_preds):
    labels_count = np.bincount(all_labels)
    prox = calc_PCNL(labels_count)
    OWE_worst = calc_ROWSS_worst(all_labels, prox)
    return 1-math.sqrt(sum([(prox[pred]-prox[label])**2/labels_count[label] for label, pred in zip(all_labels, all_preds)]))/OWE_worst


# In[9]:


def make_folded_df(df, cv):
    df["kfold"] = np.nan
    df = df.rename(columns={'Score': 'target'})
    label = df["target"].tolist()
    for fold, (_, valid_indexes) in enumerate(cv):
        df.iloc[valid_indexes, df.columns.get_loc('kfold')] = fold
    return df

def make_dataset(df, tokenizer, device):
    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(example["Text"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=MAX_LEN))
    dataset.set_format(type='torch', 
                       columns=['input_ids'
                                #, 'token_type_ids'
                                , 'attention_mask'
                                , 'target'], 
                       device=device)
    return dataset

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.config = AutoConfig.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size, num_classes)
        nn.init.normal_(self.linear.weight, std=0.03)
        nn.init.ones_(self.linear.bias)
        #nn.init.zeros_(self.linear.bias)

    def forward(self, input_ids, attention_mask):
        output, _ = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.linear(output)
        return output 
    

def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_outputs = []
    all_labels_torch = torch.empty(0).to(DEVICE)
    all_outputs_torch = torch.empty(0).to(DEVICE)

    progress = tqdm(dataloader, total=len(dataloader))

    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch+1}")

        attention_mask, input_ids, labels = batch.values()
        del batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        del input_ids, attention_mask
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        del loss

        all_labels += labels.tolist()
        all_outputs += outputs.tolist()
        all_labels_torch = torch.cat((all_labels_torch, labels), 0)
        all_outputs_torch = torch.cat((all_outputs_torch, outputs), 0)

    train_loss = total_loss / len(dataloader)

    return train_loss

def eval_fn(dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []
    all_outputs = []
    all_labels_torch = torch.empty(0).to(DEVICE)
    all_outputs_torch = torch.empty(0).to(DEVICE)

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))
        
        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch+1}")

            attention_mask, input_ids, labels = batch.values()
            del batch

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            del input_ids, attention_mask
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            del loss

            all_labels += labels.tolist()
            all_outputs += outputs.tolist()
            all_preds += preds.tolist()
            all_labels_torch = torch.cat((all_labels_torch, labels), 0)
            all_outputs_torch = torch.cat((all_outputs_torch, outputs), 0)
            
            del labels, outputs, preds

    valid_loss = total_loss / len(dataloader)
    QWK_score = cohen_kappa_score(all_labels, all_preds, labels=None, weights='quadratic', sample_weight=None)
    CEM_score = CEM(all_labels, all_preds)
    OWAS_score = OWAS_from_pred(all_labels, all_preds)
    ROWSS_score = ROWSS_from_pred(all_labels, all_preds)

    return valid_loss, QWK_score, CEM_score, OWAS_score, ROWSS_score, all_preds

def trainer(fold, df):

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = make_dataset(train_df, tokenizer, DEVICE)
    valid_dataset = make_dataset(valid_df, tokenizer, DEVICE)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
    )

    model = Classifier(MODEL_NAME)
    model = model.to(DEVICE)

    weight = torch.from_numpy(compute_class_weight('balanced',np.unique(df['target']),df['target'].values)).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)

    train_losses = []
    valid_losses = []
    valid_preds = []
    QWK_scores = []
    CEM_scores = []
    OWAS_scores = []
    ROWSS_scores = []
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_dataloader, model, criterion, optimizer, scheduler, DEVICE, epoch)
        valid_loss, QWK_score, CEM_score, OWAS_score, ROWSS_score, valid_pred = eval_fn(valid_dataloader, model, criterion, DEVICE, epoch)
        print(f"Loss: {valid_loss}", end="\n")
        print(f"QWK: {QWK_score}", end="\n")
        print(f"CEM: {CEM_score}", end="\n")
        print(f"OWAS: {OWAS_score}", end="\n")        
        print(f"ROWSS: {ROWSS_score}", end="\n")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        QWK_scores.append(QWK_score)
        CEM_scores.append(CEM_score)
        OWAS_scores.append(OWAS_score)
        ROWSS_scores.append(ROWSS_score)
        valid_preds.append(valid_pred)
        print("\n")

    return valid_losses, QWK_scores, CEM_scores, OWAS_scores, ROWSS_scores, valid_preds


# In[10]:


import pickle
df = make_folded_df(train, cv)
losses_list = []
QWK_scores_list = []
CEM_scores_list = []
OWAS_scores_list = []
ROWSS_scores_list = []
valid_preds_list = []

for fold in range(NUM_SPLITS):
    print(f"fold {fold}", "="*80)
    valid_losses, QWK_scores, CEM_scores, OWAS_scores, ROWSS_scores, valid_preds = trainer(fold, df)
    losses_list.append(valid_losses)
    QWK_scores_list.append(QWK_scores)
    CEM_scores_list.append(CEM_scores)
    OWAS_scores_list.append(OWAS_scores)
    ROWSS_scores_list.append(ROWSS_scores)
    valid_preds_list.append(valid_preds)


# In[11]:


import pickle
f = open('CSCEL_losses.txt', 'wb')
pickle.dump(losses_list, f)
f.close()
f = open('CSCEL_QWKs.txt', 'wb')
pickle.dump(QWK_scores_list, f)
f.close()
f = open('CSCEL_CEMs.txt', 'wb')
pickle.dump(CEM_scores_list, f)
f.close()
f = open('CSCEL_OWASs.txt', 'wb')
pickle.dump(OWAS_scores_list, f)
f.close()
f = open('CSCEL_ROWSSs.txt', 'wb')
pickle.dump(ROWSS_scores_list, f)
f.close()
f = open('CSCEL_valid_preds.txt', 'wb')
pickle.dump(valid_preds_list, f)
f.close()

