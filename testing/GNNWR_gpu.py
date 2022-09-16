import os
import random
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
import statsmodels.api as sm


class MYDataset(Dataset):
    def __init__(self, df, varNum, gpu):
        self.df = df
        self.gpu = gpu
        self.images = df.iloc[:,varNum+1:].values
        self.coef = df.iloc[:,1:varNum+1].values
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        coef = self.coef[idx]
        
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        coef = torch.tensor(coef, dtype=torch.float)

        return image, coef, label

class SWNN(nn.Module):
    def __init__(self, insize, outsize):
        super(SWNN, self).__init__()
        self.insize = insize
        self.outsize = outsize
        self.thissize = 0
        self.lastsize = 512
        i = 2

        self.fc = nn.Sequential()
        self.fc.add_module("full"+str(1), nn.Linear(self.insize, 512))


        while math.pow(2, int(math.log2(self.lastsize))) >= max(128, outsize + 1):
            if i == 1:
                self.thissize = int(math.pow(2, int(math.log2(self.lastsize))))
            else:
                self.thissize = int(math.pow(2, int(math.log2(self.lastsize)) - 1))
            
            self.fc.add_module("full"+str(i), nn.Linear(self.lastsize, self.thissize))
            self.fc.add_module("batc"+str(i), nn.BatchNorm1d(self.thissize))
            self.fc.add_module("acti"+str(i), nn.PReLU(init=0.4))
            self.fc.add_module("drop"+str(i), nn.Dropout(0.2))

            self.lastsize = self.thissize
            i = i + 1

        self.fc.add_module("full"+str(i), nn.Linear(self.lastsize, self.outsize))
        
    def forward(self, x):
        x = self.fc(x)
        return x

def compute_distances(P, C):
    A = (P**2).sum(axis=1, keepdims=True)
    B = (C**2).sum(axis=1, keepdims=True).T
 
    return np.sqrt(A + B - 2* np.dot(P, C.T))

def process_df(my_set, varName):
    temp_df = pd.DataFrame()

    dataset = my_set.reset_index(drop=True)

    temp_df['label'] = dataset[varName[0]]
    temp_df['beta'] = np.ones(dataset.shape[0])
    temp_df[varName[1:4]] = dataset[varName[1:4]]

    cor_df = pd.DataFrame()

    lon_mean = dataset['lon'].mean()
    lon_std = dataset['lon'].std()
    lat_mean = dataset['lat'].mean()
    lat_std = dataset['lat'].std()

    dataset['lon'][dataset['lon'] < 0] = dataset['lon'][dataset['lon'] < 0].copy() + 360.0
    cor_df['xcor'] = (dataset['lon'] - lon_mean) / lon_std
    cor_df['ycor'] = (dataset['lat'] - lat_mean) / lat_std

    sample_pt = np.array([[110.0, 0.0], [290.0,0.0], [110.0, 70.0], [290.0, 70.0]])

    cor_li = cor_df.to_numpy()
    dis_li = compute_distances(cor_li, sample_pt)
    dis_df = pd.DataFrame(dis_li)
    temp_df = temp_df.join(dis_df)
    temp_df['year'] = dataset['year'].astype('float') - 2008
    temp_df['month'] = dataset['month'].astype('float') - 6

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

        train_loss += loss.item()*data.size(0)
        
    train_loss = train_loss/len(train_loader.dataset)
    print('\r Epoch: {} \tTraining Loss:   {:.6f}'.format(epoch, train_loss))

def val(epoch):
    model.eval()
    global out
    global r2
    global best_loss
    global last_min
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
        r2 = r2_score(label_li, out_li)
        if r2 > best_loss:
            best_loss = r2
            last_min = 0
            torch.save(model, "model.pkl")
        else:
            last_min = last_min + 1

        label_li = np.array(label_li).reshape(-1)
        out_li = np.array(out_li).reshape(-1)
        
        print('\r Epoch: {} \tValidation Loss: {:.6f} \tR2: {:.6f}'.format(epoch, val_loss, best_loss))

def select(data, ratio):
    year = 0
    month = 0
    tdf = pd.DataFrame(columns=data.columns)
    positive = pd.DataFrame(columns=data.columns)
    nagative = pd.DataFrame(columns=data.columns)
    index = -1
    data.sort_values(by=['year', 'month'])
    for i in tqdm(range(0,data.shape[0])):
        if not year == data.loc[i]['year'] or not month == data.loc[i]['month']:
            if not index == -1:
                tp = tdf.sample(frac=ratio)
                tn = tdf.append(tp)
                tn.drop_duplicates(keep=False, inplace=True)

                positive = positive.append(tp)
                nagative = nagative.append(tn)
            tdf = pd.DataFrame(columns=data.columns)
            year = data.loc[i]['year']
            month = data.loc[i]['month']
            index = i
        tdf.loc[i-index] = data.loc[i]
    return positive, nagative

def test():
    model.eval()
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    label_li = np.array([])
    out_li = np.array([])
    val_loss1 = 0
    val_loss2 = 0


    with torch.no_grad():
        for data, coef, label in test_loader:
            data, coef, label = data.cuda(), coef.cuda(), label.cuda()
            data,label = data.view(data.shape[0], -1), label.view(data.shape[0], -1)

            output = model(data).mul(coef)
            output = out(output)

            loss1 = criterion1(output, label)
            loss2 = criterion2(output, label)

            a = output.view(-1).cpu().detach().numpy()
            b = label.view(-1).cpu().numpy()
            out_li = np.append(out_li, a)
            label_li = np.append(label_li, b)
            

            val_loss1 += loss1.item()*data.size(0)
            val_loss2 += loss2.item()*data.size(0)

        val_loss1 = val_loss1/len(test_loader.dataset)
        val_loss2 = val_loss2/len(test_loader.dataset)
        r2 = r2_score(label_li, out_li)

        label_li = label_li * std_li[0] + mean_li[0]
        out_li = out_li * std_li[0] + mean_li[0]



        diff = np.abs(label_li - out_li)
        diff = diff / np.abs(label_li) * 100

        rmse = math.sqrt(val_loss1) * std_li[0]
        mae  = val_loss2 * std_li[0]
        mape = diff.mean()

    print(r2, rmse, mae, mape)


def full_img(path:str):
    dataset = pd.read_csv(path, encoding="utf-8")
    dataset['fCO2'] = np.ones(dataset.shape[0])

    for i in range(1, varNum, 1):
      dataset.loc[:, varName[i]] = (dataset[varName[i]].copy() - mean_li[i] + 1.0) / std_li[i]

    img_data = MYDataset(process_df(dataset, varName=varName), varNum=varNum, gpu=is_gpu)
    img_loader = DataLoader(img_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    model.eval()
    out_li = np.array([])
    with torch.no_grad():
        for data, coef, label in img_loader:
            data, coef, label = data.cuda(), coef.cuda(), label.cuda()
            data,label = data.view(data.shape[0], -1), label.view(data.shape[0], -1)

            output = model(data).mul(coef)
            output = out(output)
            a = output.view(-1).cpu().detach().numpy()

            out_li = np.append(out_li, a)
    return out_li * std_li[0] + mean_li[0]


is_gpu = None
if torch.cuda.is_available() == True:
    is_gpu = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    is_gpu = False


varName = ['fCO2', 'Chl', 'Temp', 'Salt']
varNum = len(varName)
batch_size = 256

# dataset = pd.read_csv("D://CO2_data5.csv", encoding="utf-8")
# dataset = dataset.dropna()
# dataset = dataset[dataset.index % 4 == 0]

# df0 = dataset['date'].str.split("/",expand = True)
# df0.columns = ['year', 'month', 'date']

# dataset['month'] = df0['month']
# dataset['year'] = df0['year']

# dataset = dataset.reset_index()

#test_set, train_set = select(dataset, 0.2)
# del dataset

train_set = pd.read_csv("train1.csv")
test_set = pd.read_csv("test1.csv")

#dataset = dataset[dataset.month == '7']

# train_li = random.sample([i for i in range(0, dataset.shape[0])], int(0.8 * dataset.shape[0]))
# train_li.sort()

# test_li = list(set([i for i in range(0, dataset.shape[0])]) - set(train_li))
# test_li.sort()

# train_set = dataset.iloc[train_li, :]
# test_set  = dataset.iloc[test_li,  :]

mean_li = []
std_li = []

for i in range(0, varNum, 1):
    mean_li.append(train_set[varName[i]].mean())
    std_li.append(train_set[varName[i]].std())

train_set = train_set.copy()
test_set = test_set.copy()

# for i in range(0, varNum, 1):
#     train_set.loc[:, varName[i]] = (train_set[varName[i]].copy() - mean_li[i] + 1.0) / std_li[i]
#     test_set.loc[:, varName[i]] = (test_set[varName[i]].copy() - mean_li[i] + 1.0) / std_li[i]


train_data = MYDataset(process_df(my_set=train_set, varName=varName), varNum, gpu=is_gpu)
test_data = MYDataset(process_df(my_set=test_set, varName=varName), varNum, gpu=is_gpu)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

model = torch.load("model (2).pkl")
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

relation = str()
relation = varName[0]+"~" + "+".join(varName[1:varNum])
fit=sm.formula.ols(relation,data=train_set).fit()

r2 = 0
last_min = 0
best_loss = -1
weightlist = []
temp = []
for j in fit.params:
    temp.append(j)
weightlist.append(temp)
out = nn.Linear(4, 1, bias = False)
out.weight = nn.Parameter(torch.tensor(weightlist), requires_grad=False)

if is_gpu:
    model = model.cuda()
    out = out.cuda()

for epoch in range(1, 200000+1):
    train(epoch)
    val(epoch)
    if last_min >= 2000:
        break