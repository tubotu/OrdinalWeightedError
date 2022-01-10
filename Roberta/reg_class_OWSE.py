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
train_y = train["Score"].copy()
train_X = train['Text'].copy()


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)


# In[4]:


from scipy.stats import rankdata

def greedy_predict(outputs, train_labels):
    rank = np.round(rankdata(np.array(outputs))/len(outputs), decimals=5)
    train_class_cumsum_ratio = np.round(np.cumsum(np.bincount(np.array(train_labels))/len(train_labels)), decimals=5)
    
    pred = rank.copy()
    for i, threshould in reversed(list(enumerate(train_class_cumsum_ratio[:-1]))):
        pred = np.where((pred > threshould) & (pred < i+1), i+1, pred)
    return pred.astype(int).tolist()


# In[5]:


from functools import partial
from scipy import optimize

class OptimizedRounder(object):
    def __init__(self,
                 n_classes: int = 5,
                 initial_coef = [i+0.5 for i in range(5)]):
        self.coef_ = 0
        self.n_classes = n_classes
        self.initial_coef = initial_coef

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            X_p[i] = self.n_classes - 1 
            for j in range(self.n_classes-1):
                if pred < coef[j]:
                    X_p[i] = j
                    break
        ll = cohen_kappa_score(y, X_p, labels=None, weights='quadratic', sample_weight=None)
        return -ll
    
    def _CEM_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            X_p[i] = self.n_classes - 1 
            for j in range(self.n_classes-1):
                if pred < coef[j]:
                    X_p[i] = j
                    break
        ll = CEM(y, X_p.astype(int))
        return -ll

    def fit(self, X, y, loss_func = 'qwk'):
        if loss_func == 'qwk':
            loss_partial = partial(self._kappa_loss, X=X, y=y)
        elif loss_func == 'cem':
            loss_partial = partial(self._CEM_loss, X=X, y=y)
        else:
            raise RuntimeError('loss function does not exist!')
        self.coef_ = optimize.minimize(loss_partial, self.initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            X_p[i] = self.n_classes - 1 
            for j in range(self.n_classes-1):
                if pred < coef[j]:
                    X_p[i] = j
                    break
        return X_p

    def coefficients(self):
        return self.coef_['x']


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


# In[9]:


class OrdinalWeightedError(nn.Module):
    def def_PCNL_map(self):
        PCNL_map = {i:(torch.sum(self.distribution[0:i])+torch.sum(self.distribution[1:i+1]))                              / (2*torch.sum(self.distribution)-self.distribution[0]-self.distribution[self.n_classes-1])                     for i in range(self.n_classes)}
        return PCNL_map  
    
    def __init__(self, weight, distribution) -> None:
        super().__init__()
        self.distribution = distribution
        self.n_classes = distribution.size()[0]
        self.weight_map = {i:weight_i for i,weight_i in enumerate(weight)}
        self.PCNL_map = self.def_PCNL_map()
    
    def forward(self, input, target):
        input_floor = torch.floor(input)
        input_ceil = input_floor+1
        input_PCNL_floor = torch.tensor([self.PCNL_map[x.item()] if x.item() in self.PCNL_map.keys() 
                                        else x.item()/(self.n_classes-1) 
                                        for x in input_floor]).to(DEVICE)
        input_PCNL_ceil = torch.tensor([self.PCNL_map[x.item()] if x.item() in self.PCNL_map.keys() 
                                        else x.item()/(self.n_classes-1) 
                                        for x in input_ceil]).to(DEVICE)
        input_PCNL = input_PCNL_floor+(input_PCNL_ceil-input_PCNL_floor)*(input-input_floor).to(DEVICE)
        target_PCNL = torch.tensor([self.PCNL_map[x.item()] for x in target]).to(DEVICE)
        target_weight = torch.tensor([self.weight_map[x.item()] for x in target]).to(DEVICE)
        return torch.sum((input_PCNL-target_PCNL)**2*target_weight)/torch.sum(target_weight)


# In[10]:


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
    
class Regressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.config = AutoConfig.from_pretrained(model_name)
        self.head = AttentionHead(self.config.hidden_size,self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        nn.init.normal_(self.linear.weight, std=0.03)
        nn.init.ones_(self.linear.bias)

    def forward(self, input_ids, attention_mask):
        output, _ = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        output = output[:,0,:]
        output = self.dropout(output)
        output = self.linear(output)
        return output

def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_labels_torch = torch.empty(0).to(DEVICE)
    all_preds = []
    all_outputs = torch.empty(0).to(DEVICE)

    progress = tqdm(dataloader, total=len(dataloader))

    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch+1}")

        attention_mask, input_ids, labels = batch.values()
        del batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask).reshape(-1)
        del input_ids, attention_mask
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        del loss

        all_labels += labels.tolist()
        all_labels_torch = torch.cat((all_labels_torch, labels), 0)
        all_preds += outputs.tolist()
        all_outputs = torch.cat((all_outputs, outputs), 0)
        del labels, outputs

    train_loss = criterion(all_outputs, all_labels_torch)
    return train_loss

def eval_fn(dataloader, model, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_labels_torch = torch.empty(0).to(DEVICE)
    all_preds = []
    all_outputs = []
    all_outputs_torch = torch.empty(0).to(DEVICE)

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))
        
        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch+1}")

            attention_mask, input_ids, labels = batch.values()
            del batch

            outputs = model(input_ids, attention_mask).reshape(-1)
            preds = torch.round(outputs)
            preds = torch.clamp(preds,min=0,max=criterion.n_classes-1).to(torch.int).reshape(-1)
            del input_ids, attention_mask
            loss = criterion(outputs, labels)

            del loss

            all_labels += labels.tolist()
            all_labels_torch = torch.cat((all_labels_torch, labels), 0)
            all_preds += preds.tolist()
            
            outputs_floor = torch.floor(outputs)
            outputs_ceil = outputs_floor+1
            outputs_PCNL_floor = torch.tensor([criterion.PCNL_map[x.item()] if x.item() in criterion.PCNL_map.keys() 
                                               else x.item()/(criterion.n_classes-1)
                                               for x in outputs_floor]).to(DEVICE)
            outputs_PCNL_ceil = torch.tensor([criterion.PCNL_map[x.item()] if x.item() in criterion.PCNL_map.keys() 
                                              else x.item()/(criterion.n_classes-1)
                                              for x in outputs_ceil]).to(DEVICE)
            outputs_PCNL = outputs_PCNL_floor+(outputs_PCNL_ceil-outputs_PCNL_floor)*(outputs-outputs_floor).to(DEVICE)
            outputs_PCNL = torch.clamp(outputs_PCNL, min=0, max=1)
            all_outputs += outputs.tolist()
            all_outputs_torch = torch.cat((all_outputs_torch, outputs), 0)
            del labels, outputs, preds

            progress.set_postfix()
        valid_loss = criterion(all_outputs_torch, all_labels_torch)
    
    all_preds2 = greedy_predict(all_outputs, all_labels)
    distribution = list(criterion.PCNL_map.values())
    thresholds = [i+0.5 for i in range(len(distribution)-1)]
    optR = OptimizedRounder(n_classes = len(criterion.PCNL_map.values()), initial_coef = thresholds)
    optR.fit(all_outputs, all_labels, loss_func = 'qwk')
    all_preds3_qwk = optR.predict(all_outputs, optR.coef_['x'])
    optR = OptimizedRounder(n_classes = len(criterion.PCNL_map.values()), initial_coef = thresholds)
    optR.fit(all_outputs, all_labels, loss_func = 'cem')
    all_preds3_cem = optR.predict(all_outputs, optR.coef_['x']).astype(int).tolist()
    QWK_score = cohen_kappa_score(all_labels, all_preds, labels=None, weights='quadratic', sample_weight=None)
    QWK_score2 = cohen_kappa_score(all_labels, all_preds2, labels=None, weights='quadratic', sample_weight=None)
    QWK_score3 = cohen_kappa_score(all_labels, all_preds3_qwk, labels=None, weights='quadratic', sample_weight=None)
    CEM_score = CEM(all_labels, all_preds)
    CEM_score2 = CEM(all_labels, all_preds2)
    CEM_score3 = CEM(all_labels, all_preds3_cem)
    return valid_loss, QWK_score, QWK_score2, QWK_score3, CEM_score, CEM_score2, CEM_score3, all_preds, all_outputs

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

    model = Regressor(MODEL_NAME)
    model = model.to(DEVICE)

    weight = torch.from_numpy(compute_class_weight('balanced',np.unique(df['target']),df['target'].values)).float().to(DEVICE)
    distribution = torch.from_numpy(np.array(df['target'].value_counts().sort_index())).to(DEVICE)
    criterion = OrdinalWeightedError(weight=weight, distribution=distribution)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)

    train_losses = []
    valid_losses = []
    valid_preds = []
    QWK_scores = []
    QWK_scores2 = []
    QWK_scores3 = []
    CEM_scores = []
    CEM_scores2 = []
    CEM_scores3 = []
    valid_outputs = []
 
    best_loss = np.inf
    for epoch in range(EPOCHS):
        valid_pred = np.zeros(len(valid_df))
        valid_pred_PCNL = np.zeros(len(valid_df))
        train_loss = train_fn(train_dataloader, model, criterion, optimizer, scheduler, DEVICE, epoch)
        valid_loss, QWK_score, QWK_score2, QWK_score3, CEM_score, CEM_score2, CEM_score3, valid_pred, valid_output = eval_fn(valid_dataloader, model, criterion, DEVICE, epoch)
        print(f"Loss: {valid_loss:.4f}", end=", ")
        print(f"QWK: {QWK_score:.4f}", end=", ")
        print(f"QWK2: {QWK_score2:.4f}", end=", ")
        print(f"QWK3: {QWK_score3:.4f}", end=", ")
        print(f"CEM: {CEM_score:.4f}", end=", ")
        print(f"CEM2: {CEM_score2:.4f}", end=", ")
        print(f"CEM3: {CEM_score3:.4f}", end="\n")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        QWK_scores.append(QWK_score)
        CEM_scores.append(CEM_score)
        QWK_scores2.append(QWK_score2)
        CEM_scores2.append(CEM_score2)
        QWK_scores3.append(QWK_score3)
        CEM_scores3.append(CEM_score3)
        valid_preds.append(valid_pred)
        valid_outputs.append(valid_output)

    return valid_losses, QWK_scores, QWK_scores2, QWK_scores3, CEM_scores, CEM_scores2, CEM_scores3, valid_preds, valid_outputs


# In[ ]:


df = make_folded_df(train, cv)
losses_list = []
QWK_scores_list = []
CEM_scores_list = []
QWK_scores2_list = []
CEM_scores2_list = []
QWK_scores3_list = []
CEM_scores3_list = []
valid_preds_list = []
valid_outputs_list = []

for fold in range(NUM_SPLITS):
    print(f"fold {fold}", "="*80)
    valid_losses, QWK_scores, QWK_scores2, QWK_scores3, CEM_scores, CEM_scores2, CEM_scores3, valid_preds, valid_outputs = trainer(fold, df)
    losses_list.append(valid_losses)
    QWK_scores_list.append(QWK_scores)
    QWK_scores2_list.append(QWK_scores2)
    QWK_scores3_list.append(QWK_scores3)
    CEM_scores_list.append(CEM_scores)
    CEM_scores2_list.append(CEM_scores2)
    CEM_scores3_list.append(CEM_scores3)
    valid_preds_list.append(valid_preds)
    valid_outputs_list.append(valid_outputs)


# In[12]:


import pickle
f = open('reg_class_OWSE_losses.txt', 'wb')
pickle.dump(losses_list, f)
f.close()
f = open('reg_class_OWSE_QWK.txt', 'wb')
pickle.dump(QWK_scores_list, f)
f.close()
f = open('reg_class_OWSE_CEM.txt', 'wb')
pickle.dump(CEM_scores_list, f)
f.close()
f = open('reg_class_OWSE_QWK2.txt', 'wb')
pickle.dump(QWK_scores2_list, f)
f.close()
f = open('reg_class_OWSE_CEM2.txt', 'wb')
pickle.dump(CEM_scores2_list, f)
f.close()
f = open('reg_class_OWSE_QWK3.txt', 'wb')
pickle.dump(QWK_scores3_list, f)
f.close()
f = open('reg_class_OWSE_CEM3.txt', 'wb')
pickle.dump(CEM_scores3_list, f)
f.close()
f = open('reg_class_OWSE_valid_preds.txt', 'wb')
pickle.dump(valid_preds_list, f)
f.close()
f = open('reg_class_OWSE_valid_outputs.txt', 'wb')
pickle.dump(valid_outputs_list, f)
f.close()

