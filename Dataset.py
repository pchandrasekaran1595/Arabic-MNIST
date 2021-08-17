from torch.utils.data import Dataset
from torch import LongTensor

#########################################################################################################

class DS(Dataset):
    def __init__(self, X=None, y=None, transform=None, mode="train"):
        self.mode = mode
        self.transform = transform
        self.X = X.reshape(X.shape[0], 32, 32, 1)
        if self.mode == "train" or self.mode == "valid":
            self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "valid":
            return self.transform(self.X[idx]), LongTensor(self.y[idx])
        else:
            return self.transform(self.X[idx])
        
#########################################################################################################
