import torch
from torch import nn, optim
import utils as u

#########################################################################################################

class CNN_Model(nn.Module):
    def __init__(self, filter_sizes, HL=None):
        super(CNN_Model, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("CN1", nn.Conv2d(in_channels=1, out_channels=filter_sizes[0], kernel_size=3, stride=1, padding=1))
        self.features.add_module("BN1", nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5))
        self.features.add_module("AN1", nn.ReLU())
        self.features.add_module("MP1", nn.MaxPool2d(kernel_size=2))
        self.features.add_module("CN2", nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=3, stride=1, padding=1))
        self.features.add_module("BN2", nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5))
        self.features.add_module("AN2", nn.ReLU())
        self.features.add_module("MP2", nn.MaxPool2d(kernel_size=2))
        self.features.add_module("CN3", nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=1, padding=1))
        self.features.add_module("BN3", nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5))
        self.features.add_module("AN3", nn.ReLU())
        self.features.add_module("MP3", nn.MaxPool2d(kernel_size=2))


        self.classifier = nn.Sequential()
        if HL is None:
            self.classifier.add_module("FC", nn.Linear(in_features=filter_sizes[2] * 4 * 4, out_features=28))
            self.classifier.add_module("AN", nn.LogSoftmax(dim=1))
        elif len(HL) == 1:
            self.classifier.add_module("FC1", nn.Linear(in_features=filter_sizes[2] * 4 * 4, out_features=HL[0]))
            self.classifier.add_module("AN1", nn.ReLU())
            self.classifier.add_module("FC2", nn.Linear(in_features=HL[0], out_features=28))
            self.classifier.add_module("AN2", nn.LogSoftmax(dim=1))
        elif len(HL) == 2:
            self.classifier.add_module("FC1", nn.Linear(in_features=filter_sizes[2] * 4 * 4, out_features=HL[0]))
            self.classifier.add_module("AN1", nn.ReLU())
            self.classifier.add_module("FC2", nn.Linear(in_features=HL[0], out_features=HL[1]))
            self.classifier.add_module("AN2", nn.ReLU())
            self.classifier.add_module("FC3", nn.Linear(in_features=HL[1], out_features=28))
            self.classifier.add_module("AN3", nn.LogSoftmax(dim=1))
        else:
            raise NotImplementedError("Network Width not Implemented")
    
    def getOptimizer(self, lr=1e-3, wd=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def getScheduler(self, optimizer=None, patience=None, eps=None):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

#########################################################################################################

def build_model(filter_sizes=None, HL=None):
    assert(filter_sizes is not None)
    
    torch.manual_seed(u.SEED)
    model = CNN_Model(filter_sizes=filter_sizes, HL=HL)
    return model

#########################################################################################################
