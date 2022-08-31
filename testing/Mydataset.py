import torch
from torch.utils.data import Dataset

class MYDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.images = df.iloc[:,5:].values
        self.coef = df.iloc[:,1:5].values
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