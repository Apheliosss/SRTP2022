import os
import random
from Mydataset import MYDataset
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from sklearn.metrics import r2_score
import statsmodels.api as sm
from SWNN import GNNWR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def compute_distances(P, C):
    A = (P**2).sum(axis=1, keepdims=True)
 
    B = (C**2).sum(axis=1, keepdims=True).T
 
    return np.sqrt(A + B - 2* np.dot(P, C.T))

def process_df(my_set, varName):
    temp_df = pd.DataFrame()


    dataset = my_set.reset_index(drop=True)
    ycor = dataset.lat
    #ycor = dataset.lon


    temp_df['label'] = dataset[varName[0]]

    temp_df['beta'] = np.ones(dataset.shape[0])

    temp_df[varName[1:4]] = dataset[varName[1:4]]

    alist = dataset.lon
    temp = []
    for i in alist:
        if i < 0:
            i = i+360
        temp.append(i)
    xcor = temp

    cor_df = pd.DataFrame()
    cor_df['xcor'] = xcor
    cor_df['ycor'] = ycor

    a = [[110.0, 0.0], [290.0,0.0], [110.0, 70.0], [290.0, 70.0]]
    b = np.array(a)

    cor_li = cor_df.to_numpy()
    dis_li = compute_distances(cor_li, b)
    dis_df = pd.DataFrame(dis_li)
    temp_df = temp_df.join(dis_df)

    return temp_df

def train(epoch):
    model.train()
    train_loss = 0
    global r2
    global out
    for data, coef, label in train_loader:
        data, coef, label = data.cuda(), coef.cuda(), label.cuda()
        data, label = data.view(data.shape[0], -1), label.view(data.shape[0], -1)

        optimizer.zero_grad()

        output = model(data)
        output = output.mul(coef)
        output = out(output)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        a = output.view(-1).cpu().detach().numpy()
        b = label.view(-1).cpu().numpy()
        if epoch % 100 == 0:
            r2 = r2_score(a, b)

        train_loss += loss.item()*data.size(0)
        
    train_loss = train_loss/len(train_loader.dataset)
    print('\r Epoch: {} \tTraining Loss:   {:.6f}'.format(epoch, train_loss))

def val(epoch):
    model.eval()
    global out
    global r2
    val_loss = 0

    label_li = np.array([])
    out_li = np.array([])

    with torch.no_grad():
        for data, coef, label in test_loader:
            data, coef, label = data.cuda(), coef.cuda(), label.cuda()
            data,label = data.view(data.shape[0], -1), label.view(data.shape[0], -1)

            output = model(data).mul(coef)
            output = out(output)

            loss = criterion(output, label)

            a = output.view(-1).cpu().detach().numpy()
            b = label.view(-1).cpu().numpy()
            out_li = np.append(out_li, a)
            label_li = np.append(label_li, b)
            

            val_loss += loss.item()*data.size(0)
        val_loss = val_loss/len(test_loader.dataset)
        label_li = np.array(label_li).reshape(-1)
        out_li = np.array(out_li).reshape(-1)
        if epoch % 100 == 0:
            r2 = r2_score(out_li, label_li)
        #print(out_li)
        print('\r Epoch: {} \tValidation Loss: {:.6f} \tR2: {:.6f}'.format(epoch, val_loss, r2))


varName = ['fCO2', 'Chl', 'Temp', 'Salt']

dataset = pd.read_csv("D://CO2_data3.csv", encoding="utf-8")
dataset = dataset.dropna()
dataset = dataset[dataset.index % 4 == 0]

df0 = dataset['date'].str.split("/",expand = True)
df0.columns = ['year', 'month', 'date']

dataset['month'] = df0['month']
dataset = dataset[dataset.month == '7']

train_li = random.sample([i for i in range(0, dataset.shape[0])], int(0.8 * dataset.shape[0]))
train_li.sort()

j = 0
test_li = []

for i in range(0, dataset.shape[0], 1):
    if i != train_li[j] | j >= len(train_li):
        test_li.append(i)
    else:
        j = j + 1

train_set = dataset.iloc[train_li, :]
test_set  = dataset.iloc[test_li,  :]

mean_li = []
std_li = []

for i in range(0, len(varName), 1):
    mean_li.append(train_set[varName[i]].mean())
    std_li.append(train_set[varName[i]].std())

train_set = train_set.copy()
test_set = test_set.copy()

for i in range(0, len(varName), 1):
    train_set.loc[:, varName[i]] = (train_set[varName[i]].copy() - mean_li[i] + 1.0) / std_li[i]
    test_set.loc[:, varName[i]] = (test_set[varName[i]].copy() - mean_li[i] + 1.0) / std_li[i]

train_data = MYDataset(process_df(my_set=train_set, varName=varName))
test_data = MYDataset(process_df(my_set=test_set, varName=varName))
train_loader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_data, batch_size=50, shuffle=False, num_workers=0)

model = GNNWR(outsize=4)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

r2 = 0
weightlist = []
for i in range(1,2):
    temp = []
    temp.append(-0.172075)
    temp.append(-0.175203)
    temp.append(0.294790)
    temp.append(0.385374)
    weightlist.append(temp)
out = nn.Linear(4, 1, bias = False)
out.weight = nn.Parameter(torch.tensor(weightlist), requires_grad=False)

for epoch in tqdm(range(1, 1000+1)):
    train(epoch)
    val(epoch)
