import os
import sys
import cv2
import torch
import imgaug
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imgaug import augmenters
from torch.utils.data import DataLoader as DL
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as u
from api import fit, predict
from Model import build_model
from Dataset import DS

#########################################################################################################

def get_images_and_labels_from_csv(train=True, in_kaggle=False):
    if in_kaggle:
        if train:
            images = pd.read_csv(os.path.join(u.DATA_PATH_2, "csvTrainImages 13440x1024.csv"), engine="python")
            labels = pd.read_csv(os.path.join(u.DATA_PATH_2, "csvTrainLabel 13440x1.csv"), engine="python")
        else:
            images = pd.read_csv(os.path.join(u.DATA_PATH_2, "csvTestImages 3360x1024.csv"), engine="python")
            labels = pd.read_csv(os.path.join(u.DATA_PATH_2, "csvTestLabel 3360x1.csv"), engine="python")
    else:
        if train:
            images = pd.read_csv(os.path.join(u.DATA_PATH_1, "Train Images.csv"), engine="python")
            labels = pd.read_csv(os.path.join(u.DATA_PATH_1, "Train Labels.csv"), engine="python")
        else:
            images = pd.read_csv(os.path.join(u.DATA_PATH_1, "Test Images.csv"), engine="python")
            labels = pd.read_csv(os.path.join(u.DATA_PATH_1, "Test Labels.csv"), engine="python")
    labels = np.subtract(labels, 1)
    return images.copy().values.astype("uint8"), labels.copy().values

#########################################################################################################

def build_train_and_valid_loaders(batch_size=None, augment=False, in_kaggle=False):
    images, labels = get_images_and_labels_from_csv(train=True, in_kaggle=in_kaggle)

    if augment:
        augment = get_dataset_augment(seed=u.SEED)
        images = augment(images=images)
    
    tr_images, va_images, tr_labels, va_labels = train_test_split(images, labels, 
                                                                  test_size=0.2, 
                                                                  shuffle=True, 
                                                                  random_state=u.SEED, 
                                                                  stratify=labels)
    tr_data_setup = DS(X=tr_images, y=tr_labels.reshape(-1, 1), transform=u.TRANSFORM, mode="train")
    va_data_setup = DS(X=va_images, y=va_labels.reshape(-1, 1), transform=u.TRANSFORM, mode="valid")
    tr_data = DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(u.SEED))
    va_data = DL(va_data_setup, batch_size=batch_size, shuffle=False)
    dataloaders = {"train" : tr_data, "valid" : va_data}
    return dataloaders


def test_set_accuracy(model=None, batch_size=None, in_kaggle=False):
    images, labels = get_images_and_labels_from_csv(train=False, in_kaggle=in_kaggle)
    ts_data_setup = DS(X=images, y=None, transform=u.TRANSFORM, mode="test")
    ts_data = DL(ts_data_setup, batch_size=batch_size, shuffle=False)

    y_pred = predict(model, ts_data)
    print("Test Set Accuracy : {:.5f}".format(accuracy_score(y_pred, labels)))
    u.breaker()

#########################################################################################################
def save_graphs(L, A) -> None:
    TL, VL, TA, VA = [], [], [], []

    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])
    
    x_Axis = np.arange(1, len(L)+1, 1)
    plt.figure("Graphs")
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.grid()
    plt.legend()
    plt.title("Loss Graph")
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Train")
    plt.plot(x_Axis, VA, "b", label="Valid")
    plt.grid()
    plt.legend()
    plt.title("Accuracy Graph")
    plt.savefig("./Graphs.jpg")
    plt.close("Graphs")

#########################################################################################################

def get_dataset_augment(seed=None):
    imgaug.seed(seed)
    augment = augmenters.Sequential([
        augmenters.VerticalFlip(p=0.25),
        augmenters.HorizontalFlip(p=0.25),
        augmenters.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-45, 45), seed=seed),
    ])

    return augment

#########################################################################################################

def app():
    args_1 = "--hl"
    args_2 = "--fs"
    args_3 = "--epochs"
    args_4 = "--lr"
    args_5 = "--wd"
    args_6 = "--bs"
    args_7 = "--early"
    args_8 = "--kaggle"

    epochs = 10
    HL = None
    filter_sizes = [4, 4, 4]
    lr = 1e-3
    wd = 0
    batch_size = 64
    early_stopping = 5
    in_kaggle = False

    if args_1 in sys.argv:
        if sys.argv[sys.argv.index(args_1) + 1] == "1":
            HL = [int(sys.argv[sys.argv.index(args_1) + 2])]
        elif sys.argv[sys.argv.index(args_1) + 1] == "2":
            HL = [int(sys.argv[sys.argv.index(args_1) + 2]), 
                  int(sys.argv[sys.argv.index(args_1) + 3])]
    if args_2 in sys.argv:
        filter_sizes = [int(sys.argv[sys.argv.index(args_2) + 1]), 
                        int(sys.argv[sys.argv.index(args_2) + 2]),
                        int(sys.argv[sys.argv.index(args_2) + 3])]
    if args_3 in sys.argv:
        epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        lr = float(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv:
        wd = float(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv:
        batch_size = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv:
        early_stopping = int(sys.argv[sys.argv.index(args_7) + 1])
    if args_8 in sys.argv:
        in_kaggle = True
    
    dataloaders = build_train_and_valid_loaders(batch_size=batch_size, in_kaggle=in_kaggle)
    model = build_model(filter_sizes=filter_sizes, HL=HL)
    optimizer = model.getOptimizer(lr=lr, wd=wd)

    L, A, _, _ = fit(model=model, optimizer=optimizer, scheduler=None, epochs=epochs,
                     dataloaders=dataloaders, early_stopping_patience=early_stopping, verbose=True)
    save_graphs(L, A)
    test_set_accuracy(model, batch_size, in_kaggle)
    
#########################################################################################################
